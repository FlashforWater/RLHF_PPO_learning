import torch

from rlhf.rollout import generate_with_log_probs, compute_log_probs
from rlhf.kl_penalty import compute_rewards
from rlhf.advantage import compute_gae
from rlhf.ppo_loss import compute_policy_loss, compute_value_loss

def ppo_train_step(
    prompt_ids, prompt_mask, policy_model, ref_model, value_model, reward_model, policy_optimizer, value_optimizer, max_gen_length, 
    beta, gamma, lam, epsilon, value_coeff, ppo_epochs, eos_token_id):

    # 1. rollout
    policy_model.eval()
    with torch.no_grad():
        full_ids, full_mask, old_log_probs, response_mask = generate_with_log_probs(policy_model, prompt_ids, prompt_mask, max_gen_length, eos_token_id)
        response_length = max_gen_length

    # 2. score the generate response
    with torch.no_grad():
        reward_score = reward_model(full_ids, full_mask)

        ref_log_probs = compute_log_probs(ref_model, full_ids, full_mask, response_length)

        rewards = compute_rewards(old_log_probs, ref_log_probs, reward_score, beta, response_mask)
        values = value_model(full_ids, full_mask)
        response_values = values[:, -response_length:]

        advantages, returns = compute_gae(rewards, response_values, gamma, lam, response_mask)
        adv_alive = advantages[response_mask.bool()]
        advantages = (advantages - adv_alive.mean()) / (adv_alive.std() + 1e-8)

    
    # PPO inner update loop
    policy_model.train()
    value_model.train()

    for _ in range(ppo_epochs):
        # recompute log_probs with the current policy (gradient ON)
        new_log_probs = compute_log_probs(policy_model, full_ids, full_mask, response_length)

        policy_loss = compute_policy_loss(new_log_probs, old_log_probs, advantages, epsilon, response_mask)

        new_values = value_model(full_ids, full_mask)[:, -response_length:]
        value_loss = compute_value_loss(new_values, returns, response_mask)
        total_loss = policy_loss + value_coeff * value_loss

        policy_optimizer.zero_grad()
        value_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(),1.0)
        torch.nn.utils.clip_grad_norm_(value_model.parameters(),1.0)
        policy_optimizer.step()
        value_optimizer.step()

        metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "reward_score": reward_score.mean().item(),
            "kl":(new_log_probs - ref_log_probs).mean().item(),
            "response_len": response_mask.sum(dim = 1).mean().item()
            }

    return metrics
