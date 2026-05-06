import torch
from transformers import AutoTokenizer

from models.reward_model import RewardModel
from data.data_loader import get_reward_dataloader
from rlhf.reward_trainer import train_reward_model, evaluate_reward_model
from config import ModelConfig, RewardTrainConfig
from utils import create_run_dir, save_config


def main():
    model_config  = ModelConfig()
    reward_config = RewardTrainConfig()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    run_dir = create_run_dir("reward")
    save_config(run_dir, model=model_config, reward=reward_config)
    print(f"Run directory: {run_dir}")

    # tokenizer — set pad_token explicitly (LLaMA-style tokenizers don't ship one)
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # default right-padding is fine for the reward model
    # (the model uses attention_mask to find the real last token)

    print("Building dataloaders...")
    train_loader = get_reward_dataloader(
        "train", tokenizer, reward_config.batch_size, reward_config.max_length,
        max_examples=reward_config.max_train_examples,
    )
    val_loader = get_reward_dataloader(
        "test", tokenizer, reward_config.batch_size, reward_config.max_length,
        shuffle=False,
        max_examples=reward_config.max_eval_examples,
    )

    print("Loading reward model...")
    model = RewardModel(model_config.model_name).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=reward_config.lr)

    print("Training reward model...")
    train_reward_model(
        model, train_loader, optimizer,
        num_epochs=reward_config.num_epochs,
        device=device,
        metrics_path=run_dir / "reward_metrics.csv",
    )

    print("Evaluating on validation set...")
    accuracy = evaluate_reward_model(model, val_loader, device)
    print(f"Validation accuracy: {accuracy:.4f}")

    checkpoint_path = run_dir / "reward_model.pt"
    latest_path = "reward_model.pt"
    print(f"Saving reward model to {checkpoint_path} ...")
    torch.save(model.state_dict(), checkpoint_path)
    torch.save(model.state_dict(), latest_path)
    print("Done.")


if __name__ == "__main__":
    main()
