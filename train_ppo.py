import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.reward_model import RewardModel
from models.value_model import ValueModel
from rlhf.ppo_trainer import ppo_train_step
from data.data_loader import get_ppo_dataloader
from config import ModelConfig, PPOConfig

def main():
    model_config = ModelConfig()
    ppo_config = PPOConfig()

    device = 'cuda' if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    eos_token_id = tokenizer.eos_token_id

    prompt_loader = get_ppo_dataloader(
        "train",
        tokenizer,
        ppo_config.batch_size,
        max_examples=ppo_config.max_prompt_examples,
    )

    # Load four models
    print("Loading policy model...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        dtype=torch.float32,
    ).to(device)

    print("Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        dtype=torch.float32,
    ).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    

    print("Loading reward model...")
    reward_model = RewardModel(model_config.model_name).to(device)
    reward_model.load_state_dict(torch.load("reward_model.pt", map_location = device))
    for p in reward_model.parameters():
        p.requires_grad = False
    reward_model.eval()

    print("Loading value model...")
    value_model = ValueModel(model_config.model_name).to(device)

    policy_optimizer = torch.optim.AdamW(policy_model.parameters(), lr = ppo_config.policy_lr)
    value_optimizer = torch.optim.AdamW(value_model.parameters(), lr = ppo_config.value_lr)

    print("Start PPO training...")
    iteration = 0
    while iteration < ppo_config.num_iterations:
        for batch in prompt_loader:
            if iteration >= ppo_config.num_iterations:
                break

            prompt_ids = batch['prompt_ids'].to(device)
            prompt_mask = batch['prompt_mask'].to(device)

            metrics = ppo_train_step(
                prompt_ids, prompt_mask,
                policy_model, ref_model, value_model, reward_model,
                policy_optimizer, value_optimizer,
                ppo_config.max_gen_length,
                ppo_config.beta,
                ppo_config.gamma,
                ppo_config.lam,
                ppo_config.epsilon,
                ppo_config.value_coeff,
                ppo_config.ppo_epochs,
                eos_token_id,
            )

            print(f"[{iteration}] reward = {metrics['reward_score']:.3f},"
            f"kl = {metrics['kl']:.3f},"
            f"policy_loss = {metrics['policy_loss']:.4f},"
            f"value_loss = {metrics['value_loss']:.4f}")

            iteration += 1

    torch.save(policy_model.state_dict(), "ppo_policy.pt")
    torch.save(value_model.state_dict(), "ppo_value.pt")
    print("PPO training complete. Models saved.")

if __name__ == "__main__":
    main()
