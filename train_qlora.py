import logging
import os
from datetime import datetime

import torch
import torch.nn.functional as F
import transformers

import blackjack
from grpo import train_grpo

from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

from tqdm import tqdm

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

    model_id = "ibm-granite/granite-4.0-micro"
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
        # rule of thumb: for dense rewards, use 32+ for rank. since we're doing token-level rewards, we have dense rewards
        r=32,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    policy_model = get_peft_model(policy_model, lora_config)

    base_lr = 5e-5
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

    train_grpo(
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        optim=optim,
        sched=sched,
        logger=logger,
        run_dir=run_dir,
        total_steps=total_steps,
        eval_every=eval_every,
        eval_games=eval_games,
        eval_seed=eval_seed,
        train_temperature=train_temperature,
        max_seq_len=max_seq_len,
        groups_per_batch=groups_per_batch,
        rollouts_per_group=rollouts_per_group,
        kl_coef=kl_coef,
        entropy_coef=entropy_coef,
        ppo_epochs=ppo_epochs,
        generate_trajectory=generate_trajectory,
        game_cls=blackjack.Blackjack,
    )
