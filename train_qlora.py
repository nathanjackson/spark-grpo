import copy
import random
import logging
import os
from datetime import datetime

import torch
import torch.nn.functional as F
import transformers

import blackjack

from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

from tqdm import tqdm

# Shape Constant Meanings:
# B = Batch Size
# T = Tokens

def grpo_token_level_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    mask: torch.Tensor,
    group_ids: torch.Tensor | None = None,
    kl_coef: float = 0.1,
    temp: float = 1.0,
    eps: float = 1e-8,
    reduction: str = "batch"
):
    """
    Computes the token-level GRPO loss.

    Args:
        log_probs: (B, T) token logprobs from current policy model
        old_log_probs: (B, T) token logprobs from behavior (old) policy
        ref_log_probs: (B, T) token logprobs from reference model
        rewards: (B, T) token-level rewards
        mask: (B, T) mask that indicates which tokens should contribute to the
              advantage normalization and gradient computation
        kl_coef: coefficient for KL regularizer
        temp: temperature for advantage scaling
        eps: small epsilon value to avoid divide-by-zero in normalization
        reduction: "batch" | "per_sample" for final loss aggregation
    Returns:
        loss: scalar tensor if reduction != "per_sample", else tensor of shape (B, ) with per-sample mean loss
    """
    assert 2 == log_probs.ndim
    assert 2 == mask.ndim
    assert rewards.ndim == mask.ndim
    assert log_probs.shape == old_log_probs.shape
    assert log_probs.shape == ref_log_probs.shape
    assert rewards.shape == mask.shape
    assert log_probs.shape == rewards.shape

    # don't propagate backwards through rewards, mask, and reference log probs
    rewards = rewards.detach()
    old_log_probs = old_log_probs.detach()
    ref_log_probs = ref_log_probs.detach()
    mask = mask.detach()

    # compute mean/std over trajectories (one reward per sample)
    mask_f = mask.float()
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    sample_rewards = (rewards * mask_f).sum(dim=1) / denom
    if sample_rewards.numel() == 0:
        # no valid tokens; return zero loss with grad
        return (log_probs * 0).sum()
    if group_ids is None:
        mean = sample_rewards.mean()
        std = sample_rewards.std(unbiased=False).clamp(min=eps)
        # normalized advantages (avoid zeroing; keeps gradient signal when rewards collapse)
        adv = ((sample_rewards - mean) / std).unsqueeze(1) * mask_f
    else:
        group_ids = group_ids.to(sample_rewards.device)
        global_mean = sample_rewards.mean()
        global_std = sample_rewards.std(unbiased=False).clamp(min=eps)
        means = torch.zeros_like(sample_rewards)
        stds = torch.zeros_like(sample_rewards)
        for gid in torch.unique(group_ids):
            grp_mask = group_ids == gid
            grp_rewards = sample_rewards[grp_mask]
            if grp_rewards.numel() == 0:
                continue
            grp_mean = grp_rewards.mean()
            grp_std = grp_rewards.std(unbiased=False)
            # If within-group variance collapses, fall back to global normalization
            if grp_rewards.numel() < 2 or grp_std < 1e-6:
                means[grp_mask] = global_mean
                stds[grp_mask] = global_std
            else:
                means[grp_mask] = grp_mean
                stds[grp_mask] = grp_std.clamp(min=eps)
        adv = ((sample_rewards - means) / stds).unsqueeze(1) * mask_f

    # importance weights
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * adv
    #w = torch.exp(log_probs - ref_log_probs)

    # policy loss per token
    policy_loss = -1. * torch.minimum(surr1, surr2) * mask
    #policy_loss = -(w * adv) * mask

    # KL term
    policy_probs = log_probs.exp()
    ref_probs = ref_log_probs.exp()
    kl_per_token = ref_probs * (ref_log_probs - log_probs)
    kl_per_token = kl_per_token * mask

    loss = policy_loss + kl_coef * kl_per_token.mean()

    if reduction == "batch":
        return (loss.sum(-1) / mask.sum(-1)).mean()
    elif reduction == "per_sample":
        return loss.sum(dim=1) / mask.sum(dim=1)
    else:
        raise ValueError(f"unknown reduction: {reduction}")

def _build_action_token_sequences(tokenizer):
    # Include common variants with/without leading space.
    variants = ["HIT", " HIT", "STAY", " STAY"]
    sequences = []
    for text in variants:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            sequences.append(ids)
    # Deduplicate while preserving order.
    deduped = []
    for seq in sequences:
        if seq not in deduped:
            deduped.append(seq)
    return deduped


def _make_prefix_allowed_tokens_fn(prompt_len, action_seqs, eos_token_id):
    def prefix_allowed_tokens_fn(batch_id, input_ids):
        # Only constrain the generated portion after the prompt.
        gen = input_ids[prompt_len:].tolist()
        # If nothing generated yet, allow the first token of any action.
        if len(gen) == 0:
            return sorted({seq[0] for seq in action_seqs if len(seq) > 0})
        allowed = set()
        for seq in action_seqs:
            if gen == seq:
                # Allow EOS (or any token that ends generation) once full action is matched.
                if eos_token_id is not None:
                    allowed.add(eos_token_id)
                continue
            if len(gen) < len(seq) and gen == seq[:len(gen)]:
                allowed.add(seq[len(gen)])
        return sorted(allowed) if allowed else ([] if eos_token_id is None else [eos_token_id])
    return prefix_allowed_tokens_fn


def generate_trajectory(game, tokenizer, model, temperature: float = 1.0):
    device = model.device
    messages = [
        { "role": "system", "content": ("You are playing blackjack. At each "
            "step you'll be given the state of the game. Respond with HIT or "
            "STAY. Do not add commentary.")
        }
    ]
    action_mask = None
    token_rewards = None

    invalid_action = False
    sequence_ids = None
    #print("start game")
    while not game.get_state()["game_over"]:
        visible_state = {
            "your_hand": game.get_state()["player_hand"],
            "dealer_hand": ["Hidden"] + game.get_state()["dealer_hand"][1:]
        }
        state_str = str(visible_state)
        #print(state_str)
        messages.append({ "role": "user", "content": state_str })

        messages_text = tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True)
        #print(messages_text)
        messages_encoding = tokenizer(messages_text, return_tensors="pt").to(device)

        prompt_len = messages_encoding.input_ids.shape[1]
        action_seqs = _build_action_token_sequences(tokenizer)
        prefix_allowed_tokens_fn = _make_prefix_allowed_tokens_fn(
            prompt_len,
            action_seqs,
            tokenizer.eos_token_id,
        )
        with torch.no_grad():
            outputs = model.generate(
                **messages_encoding,
                max_new_tokens=4,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                return_dict_in_generate=True,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )
        sequence_ids = outputs.sequences[0]
        action_ids = outputs.sequences[0, prompt_len:].unsqueeze(0)
        if action_mask is None:
            action_mask = torch.zeros(outputs.sequences.shape[1], dtype=torch.bool).to(device)
        else:
            new_ids = outputs.sequences[0, action_mask.shape[0]:]
            action_mask = torch.cat((action_mask, torch.zeros(new_ids.shape[0], dtype=torch.bool).to(device)), dim=0)
        response_ids = outputs.sequences[0, prompt_len:]
        if token_rewards is None:
            token_rewards = torch.zeros(outputs.sequences.shape[1], dtype=torch.float32, device=device)
        else:
            new_ids = outputs.sequences[0, token_rewards.shape[0]:]
            token_rewards = torch.cat((token_rewards, torch.zeros(new_ids.shape[0], dtype=torch.float32, device=device)), dim=0)
        #action_mask[messages_encoding.input_ids.shape[1]:][response_ids != tokenizer.eos_token_id] = 1
        #print(action_mask)
        action = tokenizer.batch_decode(action_ids, skip_special_tokens=True)[0]
        action = action.strip().upper()
        #print(f"Action: {action}")
        #print(f"Action token IDs: {action_ids[0].tolist()}")
        #print(f"Mask for these tokens: {action_mask[messages_encoding.input_ids.shape[1]:messages_encoding.input_ids.shape[1]+len(action_ids[0])].tolist()}")
        messages.append({ "role": "assistant", "content": action})
        #print(action)

        if action == "HIT":
            #print("model hits")
            observation = game.hit()
            action_mask[messages_encoding.input_ids.shape[1]:][response_ids != tokenizer.eos_token_id] = 1
        elif action == "STAY":
            #print("model stays")
            observation = game.stand()
            action_mask[messages_encoding.input_ids.shape[1]:][response_ids != tokenizer.eos_token_id] = 1
        else:
            #print("invalid action")
            invalid_action = True
            action_mask[messages_encoding.input_ids.shape[1]:][response_ids != tokenizer.eos_token_id] = 1
            #action_mask[messages_encoding.input_ids.shape[1]:] = 1
            break

    if invalid_action:
        # agent did not play properly
        #print("game did not finish")
        reward = -2.
    elif "push" in observation.lower():
        #print("push")
        reward = 1. 
    elif "player wins" in observation.lower():
        #print("agent win")
        reward = 2.
    else:
        #print("agent loss")
        reward = -1.

    if token_rewards is None:
        token_rewards = torch.zeros(sequence_ids.shape[0], dtype=torch.float32, device=device)
    token_rewards[action_mask] += reward

    #print("Reward:", reward)
    #print("Mask:", action_mask)

    with torch.no_grad():
        logits = model(sequence_ids.unsqueeze(0)).logits
        logprobs = F.log_softmax(logits, dim=-1)
        old_logprobs = logprobs[:, :-1, :].gather(
            -1, sequence_ids.unsqueeze(0)[:, 1:].unsqueeze(-1)
        ).squeeze(-1).squeeze(0)

    return {
        "messages": messages,
        "sequence_ids": sequence_ids,
        "action_mask": action_mask,
        "old_logprobs": old_logprobs,
        "token_rewards": token_rewards,
        "reward": reward
    }


if "__main__" == __name__:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"grpo_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    log_filename = os.path.join(run_dir, "grpo.log")
    log_handlers = [
        logging.FileHandler(log_filename),
        logging.StreamHandler(),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=log_handlers,
    )
    logger = logging.getLogger("grpo")

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model_id = "ibm-granite/granite-3.3-8b-instruct"
    #model_id = "Qwen/Qwen2-0.5B-Instruct"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    ref_model = transformers.AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config).to("cuda")
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False  # Freeze all parameters
    policy_model = transformers.AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config).to("cuda")
    policy_model.gradient_checkpointing_enable()
    policy_model = prepare_model_for_kbit_training(policy_model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    policy_model = get_peft_model(policy_model, lora_config)

    base_lr = 5e-6
    total_steps = 10000
    warmup_steps = 20
    optim = torch.optim.AdamW(policy_model.parameters(), lr=base_lr, fused=True)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optim,
        start_factor=1.0 / max(1, warmup_steps),
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        T_max=max(1, total_steps - warmup_steps),
    )
    sched = torch.optim.lr_scheduler.SequentialLR(
        optim,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_steps],
    )

    eval_every = 100
    eval_games = 1000
    eval_seed = 1337
    train_temperature = 1.6
    max_seq_len = 384
    groups_per_batch = 2
    rollouts_per_group = 6
    kl_coef = 0.2
    entropy_coef = 0.03
    ppo_epochs = 2

    def run_eval(step_label):
        py_state = random.getstate()
        random.seed(eval_seed)
        policy_model.eval()
        wins = 0
        losses = 0
        pushes = 0
        with torch.no_grad():
            for _ in tqdm(range(eval_games)):
                eval_game = blackjack.Blackjack()
                eval_game.start_round()
                eval_traj = generate_trajectory(
                    eval_game,
                    tokenizer,
                    policy_model,
                    temperature=1.0,
                )
                if eval_traj["reward"] == 2.0:
                    wins += 1
                elif eval_traj["reward"] == 1.0:
                    pushes += 1
                else:
                    losses += 1
        total = wins + losses + pushes
        if total > 0:
            win_rate = wins / total
            logger.info(
                "[eval] step: %s\twin_rate: %.3f\twins: %s\tpushes: %s\tlosses: %s",
                step_label,
                win_rate,
                wins,
                pushes,
                losses,
            )
        checkpoint_dir = os.path.join(run_dir, f"checkpoint_{step_label}")
        policy_model.save_pretrained(checkpoint_dir)
        random.setstate(py_state)

    run_eval("init")
    for step in range(total_steps):
        batch_ids = None
        batch_rewards = None
        batch_actions = None
        batch_old_logprobs = None
        batch_group_ids = []
        policy_model.eval()
        for group_idx in range(groups_per_batch):
            game = blackjack.Blackjack()
            game.start_round()
            for _ in range(rollouts_per_group):
                current_game = copy.deepcopy(game)
                trajectory = generate_trajectory(
                    current_game,
                    tokenizer,
                    policy_model,
                    temperature=train_temperature,
                )
                batch_group_ids.append(group_idx)
                #print(f"action_mask from generation length: {trajectory['action_mask'].shape[0]}")
                #print(f"sequence_ids length: {trajectory['sequence_ids'].shape[0]}")
    
                #sequence_text = tokenizer.apply_chat_template(trajectory["messages"], tokenize=False)[:-1]
                #print(sequence_text)
                #sequence_encoding = tokenizer(sequence_text, return_tensors="pt", padding="max_length", max_length=512).to(ref_model.device)
                #print(f"re-tokenized non-pad length: {(sequence_encoding.input_ids != tokenizer.pad_token_id).sum().item()}")
                sequence_ids = trajectory["sequence_ids"]
                if sequence_ids.shape[0] > max_seq_len:
                    sequence_ids = sequence_ids[-max_seq_len:]
                #print("sequence ids shape:", sequence_ids.shape)
                sequence_padding = torch.full((max_seq_len - sequence_ids.shape[0], ), tokenizer.pad_token_id, dtype=torch.long).to(ref_model.device)
                sequence_ids = torch.cat((sequence_ids, sequence_padding)).unsqueeze(0)
                #print("sequence ids shape:", sequence_ids.shape)

                if batch_ids is None:
                    batch_ids = sequence_ids
                else:
                    batch_ids = torch.cat((batch_ids, sequence_ids))

                rewards = trajectory["token_rewards"]
                if rewards.shape[0] > max_seq_len:
                    rewards = rewards[-max_seq_len:]
                reward_padding = torch.zeros((max_seq_len - rewards.shape[0], ), dtype=rewards.dtype).to(ref_model.device)
                rewards = torch.cat((rewards, reward_padding)).unsqueeze(0)
                if batch_rewards is None:
                    batch_rewards = rewards
                else:
                    batch_rewards = torch.cat((batch_rewards, rewards))

                #print("actions mask shape:", trajectory["action_mask"].shape)
                action_mask = trajectory["action_mask"]
                if action_mask.shape[0] > max_seq_len:
                    action_mask = action_mask[-max_seq_len:]
                action_padding = torch.zeros((max_seq_len - action_mask.shape[0], ), dtype=torch.bool).to(ref_model.device)
                action_mask = torch.cat((action_mask, action_padding))

                #print(f"After padding - action_mask shape: {action_mask.shape}")
                #print(f"After padding - action_mask sum: {action_mask.sum()}")
                #print(f"sequence_ids shape: {sequence_ids.shape}")
                #print(f"Pad token positions: {(sequence_ids == tokenizer.pad_token_id).sum()}")

                #token_list = tokenizer.convert_ids_to_tokens(sequence_ids[0].tolist())
                #for idx in ((action_mask == 1).nonzero().squeeze(1).tolist()):
                #    print(idx)
                #    print(token_list[idx])

                action_mask = action_mask.unsqueeze(0)

                #print("action mask:", action_mask)
                if batch_actions is None:
                    batch_actions = action_mask
                else:
                    batch_actions = torch.cat((batch_actions, action_mask))
                #print("batch actions shape:", batch_actions.shape)

                old_lp = trajectory["old_logprobs"]
                if old_lp.shape[0] > (max_seq_len - 1):
                    old_lp = old_lp[-(max_seq_len - 1):]
                old_lp_padding = torch.full((max_seq_len - 1 - old_lp.shape[0], ), 0.0, dtype=old_lp.dtype).to(ref_model.device)
                old_lp = torch.cat((old_lp, old_lp_padding)).unsqueeze(0)
                if batch_old_logprobs is None:
                    batch_old_logprobs = old_lp
                else:
                    batch_old_logprobs = torch.cat((batch_old_logprobs, old_lp))

        ref_model.eval()
        # ref_logprobs from a frozen reference model
        with torch.no_grad():
            ref_logits = ref_model(batch_ids).logits
            ref_logprobs = F.log_softmax(ref_logits, dim=-1)
            ref_logprobs = ref_logprobs[:, :-1, :].gather(
                -1, batch_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)

        #print(f"batch_actions shape before shift: {batch_actions.shape}")    
        batch_actions = batch_actions[:, 1:]
        batch_rewards = batch_rewards[:, 1:]
        #print(f"batch_actions shape after shift: {batch_actions.shape}")
        #print(f"batch_actions sum: {batch_actions.sum()}")
        #print(f"batch_rewards unique: {batch_rewards[batch_actions].unique()}")
        #print(f"First sample action positions: {batch_actions[0].nonzero().squeeze()}")
        #print(f"batch_actions shape: {batch_actions.shape}")
        #print(f"batch_actions sum per sample: {batch_actions.sum(dim=1)}")
        #print(f"batch_rewards shape: {batch_rewards.shape}")

        #IPython.embed()
        # distribute reward to actions
        valid_rewards = batch_rewards[batch_actions]
        if valid_rewards.numel() == 0:
            reward_mean = torch.tensor(0.0, device=valid_rewards.device)
            reward_std = torch.tensor(0.0, device=valid_rewards.device)
        else:
            reward_mean = valid_rewards.mean()
            reward_std = valid_rewards.std()
        #batch_rewards = batch_actions * (batch_rewards[:,0] / batch_actions.sum(dim=1)).unsqueeze(0).T

        #valid_rewards = batch_rewards[batch_actions]
        #print(f"Valid rewards: {valid_rewards}")
        #print(f"Unique rewards: {valid_rewards.unique()}")
        #print(f"Reward mean: {valid_rewards.mean()}")
        #print(f"Reward std: {valid_rewards.std()}")
        # KL Coef for stability
        group_ids = torch.tensor(batch_group_ids, device=batch_ids.device, dtype=torch.long)
        last_loss = None
        last_grad_norm = None
        last_policy_logprobs = None
        for _ in range(ppo_epochs):
            policy_model.eval()
            policy_logits = policy_model(batch_ids).logits
            policy_logprobs_full = F.log_softmax(policy_logits, dim=-1)
            policy_logprobs = policy_logprobs_full[:, :-1, :].gather(
                -1, batch_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)

            loss = grpo_token_level_loss(
                policy_logprobs,
                batch_old_logprobs,
                ref_logprobs,
                batch_rewards,
                batch_actions,
                group_ids=group_ids,
                kl_coef=kl_coef,
            )
            entropy = -(policy_logprobs_full.exp() * policy_logprobs_full).sum(dim=-1)[:, :-1]
            entropy_denom = batch_actions.sum(dim=1).clamp_min(1.0)
            entropy_mean = (entropy * batch_actions).sum(dim=1) / entropy_denom
            loss = loss - entropy_coef * entropy_mean.mean()

            policy_model.train()
            optim.zero_grad()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
            if grad_norm == 0.0 and loss.item() != 0.0:
                valid_action_count = batch_actions.sum().item()
                any_grad = any(
                    (p.grad is not None) and torch.any(p.grad != 0).item()
                    for p in policy_model.parameters()
                )
                logger.warning(
                    "[debug] zero grad_norm with nonzero loss"
                    "\tloss_requires_grad: %s"
                    "\tpolicy_logprobs_requires_grad: %s"
                    "\tvalid_action_tokens: %s"
                    "\tany_nonzero_grad: %s",
                    loss.requires_grad,
                    policy_logprobs.requires_grad,
                    int(valid_action_count),
                    any_grad,
                )
            optim.step()
            last_loss = loss
            last_grad_norm = grad_norm
            last_policy_logprobs = policy_logprobs.detach()

        lr = None
        for param_group in optim.param_groups:
            lr = param_group['lr']

        # Debug stats to track collapse
        mask_f = batch_actions.float()
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        sample_rewards = (batch_rewards * mask_f).sum(dim=1) / denom
        sample_rewards_std = sample_rewards.std(unbiased=False)
        sample_rewards_mean = sample_rewards.mean()
        adv = ((sample_rewards - sample_rewards_mean) / sample_rewards_std.clamp(min=1e-8)).unsqueeze(1) * mask_f
        adv_abs_mean = adv.abs().mean()
        policy_logprobs = last_policy_logprobs
        loss = last_loss
        grad_norm = last_grad_norm
        ratio = torch.exp(policy_logprobs - batch_old_logprobs)
        if batch_actions.any():
            ratio_vals = ratio[batch_actions]
            ratio_mean = ratio_vals.mean()
            ratio_std = ratio_vals.std(unbiased=False)
            valid_action_tokens = int(batch_actions.sum().item())
        else:
            ratio_mean = torch.tensor(0.0, device=policy_logprobs.device)
            ratio_std = torch.tensor(0.0, device=policy_logprobs.device)
            valid_action_tokens = 0
        logger.info(
            "step: %s\tloss: %.20f\tgrad_norm: %.8f\tlr: %s\treward_mean: %.4f\treward_std: %.4f"
            "\tsample_reward_std: %.6f\tadv_abs_mean: %.6f\tratio_mean: %.6f\tratio_std: %.6f\tvalid_action_tokens: %s",
            step,
            loss.item(),
            grad_norm,
            lr,
            reward_mean.item(),
            reward_std.item(),
            sample_rewards_std.item(),
            adv_abs_mean.item(),
            ratio_mean.item(),
            ratio_std.item(),
            valid_action_tokens,
        )

        sched.step()

        if step > 0 and step % eval_every == 0:
            run_eval(step)

        #if grad_norm > 50.:
        #    break
