import torch


def compute_gae(rewards, values, gamma, lam, mask=None):
    if mask is None:
        mask = torch.ones_like(rewards)

    seq_len = rewards.shape[1]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(rewards.size(0), device=rewards.device, dtype=rewards.dtype)

    for t in range(seq_len - 1, -1, -1):
        if t == seq_len - 1:
            next_values = torch.zeros_like(values[:, t])
            next_mask = torch.zeros_like(mask[:, t])
        else:
            next_values = values[:, t + 1]
            next_mask = mask[:, t + 1]

        delta = rewards[:, t] + gamma * next_values * next_mask - values[:, t]
        last_gae = delta + gamma * lam * next_mask * last_gae
        advantages[:, t] = last_gae

    advantages = advantages * mask
    returns = advantages + values
    return advantages, returns
