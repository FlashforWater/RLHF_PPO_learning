import torch
from transformers import AutoTokenizer

from config import ModelConfig, RewardTrainConfig
from data.data_loader import get_reward_dataloader
from models.reward_model import RewardModel


def main():
    model_config = ModelConfig()
    reward_config = RewardTrainConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    loader = get_reward_dataloader(
        "test",
        tokenizer,
        reward_config.batch_size,
        reward_config.max_length,
        shuffle=False,
        max_examples=reward_config.max_eval_examples,
    )

    model = RewardModel(model_config.model_name).to(device)
    model.load_state_dict(torch.load("reward_model.pt", map_location=device))
    model.eval()

    correct = 0
    total = 0
    chosen_sum = 0.0
    rejected_sum = 0.0

    with torch.no_grad():
        for batch in loader:
            chosen_ids = batch["chosen_ids"].to(device)
            chosen_mask = batch["chosen_mask"].to(device)
            rejected_ids = batch["rejected_ids"].to(device)
            rejected_mask = batch["rejected_mask"].to(device)

            chosen_reward = model(chosen_ids, chosen_mask)
            rejected_reward = model(rejected_ids, rejected_mask)

            correct += (chosen_reward > rejected_reward).sum().item()
            total += chosen_reward.size(0)
            chosen_sum += chosen_reward.sum().item()
            rejected_sum += rejected_reward.sum().item()

    accuracy = correct / max(total, 1)
    chosen_mean = chosen_sum / max(total, 1)
    rejected_mean = rejected_sum / max(total, 1)
    print(f"accuracy: {accuracy:.4f}")
    print(f"chosen_reward_mean: {chosen_mean:.4f}")
    print(f"rejected_reward_mean: {rejected_mean:.4f}")
    print(f"reward_gap: {chosen_mean - rejected_mean:.4f}")


if __name__ == "__main__":
    main()
