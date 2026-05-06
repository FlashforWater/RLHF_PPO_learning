import torch

def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum().clamp(min = 1.0)


def compute_policy_loss(log_probs, old_log_probs, advantages, epsilon, mask):
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    loss_per_token = -torch.min(surr1, surr2)
    return masked_mean(loss_per_token, mask)

def compute_value_loss(values, returns, mask):
    loss_per_token =((values - returns) ** 2)
    return masked_mean(loss_per_token, mask)

def compute_ppo_loss(log_probs, old_log_probs, advantages, values, returns, epsilon, value_coeff, mask):
    policy_loss = compute_policy_loss(log_probs, old_log_probs, advantages, epsilon, mask)
    value_loss = compute_value_loss(values, returns, mask)
    total_loss = policy_loss + value_coeff * value_loss
    return total_loss, policy_loss, value_loss