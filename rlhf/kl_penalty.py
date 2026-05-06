import torch
def compute_kl_penalty(log_probs, ref_log_probs, beta):
    kl = log_probs - ref_log_probs
    penalty = - beta * kl
    return penalty

def compute_rewards(log_probs, ref_log_probs, reward_score, beta, response_mask):
    rewards = compute_kl_penalty(log_probs, ref_log_probs, beta)
    rewards = rewards * response_mask

    last_alive_idx = response_mask.long().sum(dim = 1) - 1
    batch_idx = torch.arange(rewards.size(0), device = rewards.device)

    rewards[batch_idx, last_alive_idx] += reward_score.squeeze(-1)
    return rewards 





