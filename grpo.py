import torch


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
