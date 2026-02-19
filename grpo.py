import copy
import os
import random

import torch
import torch.nn.functional as F
import tqdm


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
    reduction: str = "batch",
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


def train_grpo(
    *,
    policy_model,
    ref_model,
    tokenizer,
    optim,
    sched,
    logger,
    run_dir,
    total_steps,
    eval_every,
    eval_games,
    eval_seed,
    train_temperature,
    max_seq_len,
    groups_per_batch,
    rollouts_per_group,
    kl_coef,
    entropy_coef,
    ppo_epochs,
    generate_trajectory,
    game_cls,
    eval_iter_factory=None,
):
    if eval_iter_factory is None:
        eval_iter_factory = range

    def run_eval(step_label, eval_model):
        py_state = random.getstate()
        random.seed(eval_seed)
        policy_model.eval()
        wins = 0
        losses = 0
        pushes = 0
        with torch.no_grad():
            for _ in tqdm.tqdm(eval_iter_factory(eval_games)):
                eval_game = game_cls()
                eval_game.start_round()
                eval_traj = generate_trajectory(
                    eval_game,
                    tokenizer,
                    eval_model,
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
        eval_model.save_pretrained(checkpoint_dir)
        random.setstate(py_state)

    run_eval("init", ref_model)
    for step in range(total_steps):
        batch_ids = None
        batch_rewards = None
        batch_actions = None
        batch_old_logprobs = None
        batch_group_ids = []
        policy_model.eval()
        for group_idx in range(groups_per_batch):
            game = game_cls()
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
                sequence_padding = torch.full(
                    (max_seq_len - sequence_ids.shape[0],),
                    tokenizer.pad_token_id,
                    dtype=torch.long,
                ).to(ref_model.device)
                sequence_ids = torch.cat((sequence_ids, sequence_padding)).unsqueeze(0)
                #print("sequence ids shape:", sequence_ids.shape)

                if batch_ids is None:
                    batch_ids = sequence_ids
                else:
                    batch_ids = torch.cat((batch_ids, sequence_ids))

                rewards = trajectory["token_rewards"]
                if rewards.shape[0] > max_seq_len:
                    rewards = rewards[-max_seq_len:]
                reward_padding = torch.zeros(
                    (max_seq_len - rewards.shape[0],),
                    dtype=rewards.dtype,
                ).to(ref_model.device)
                rewards = torch.cat((rewards, reward_padding)).unsqueeze(0)
                if batch_rewards is None:
                    batch_rewards = rewards
                else:
                    batch_rewards = torch.cat((batch_rewards, rewards))

                #print("actions mask shape:", trajectory["action_mask"].shape)
                action_mask = trajectory["action_mask"]
                if action_mask.shape[0] > max_seq_len:
                    action_mask = action_mask[-max_seq_len:]
                action_padding = torch.zeros(
                    (max_seq_len - action_mask.shape[0],),
                    dtype=torch.bool,
                ).to(ref_model.device)
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
                old_lp_padding = torch.full(
                    (max_seq_len - 1 - old_lp.shape[0],),
                    0.0,
                    dtype=old_lp.dtype,
                ).to(ref_model.device)
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
        group_ids = torch.tensor(
            batch_group_ids,
            device=batch_ids.device,
            dtype=torch.long,
        )
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

            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy_model.parameters(),
                max_norm=1.0,
            )
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
            lr = param_group["lr"]

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
            run_eval(step, policy_model)

        #if grad_norm > 50.:
        #    break
