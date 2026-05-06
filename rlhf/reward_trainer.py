import torch
from utils import append_csv_row


def compute_reward_loss(reward_chosen, reward_rejected):
    """Bradley-Terry loss: push chosen reward above rejected reward."""
    loss = -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()
    return loss


def train_reward_model(model, train_loader, optimizer, num_epochs, device, metrics_path=None):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            print(f"  starting batch {num_batches + 1}/{len(train_loader)}", flush=True)
            chosen_ids    = batch["chosen_ids"].to(device)
            chosen_mask   = batch["chosen_mask"].to(device)
            rejected_ids  = batch["rejected_ids"].to(device)
            rejected_mask = batch["rejected_mask"].to(device)

            reward_chosen   = model(chosen_ids, chosen_mask)
            reward_rejected = model(rejected_ids, rejected_mask)
            loss = compute_reward_loss(reward_chosen, reward_rejected)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            if metrics_path is not None:
                append_csv_row(
                    metrics_path,
                    {
                        "epoch": epoch + 1,
                        "batch": num_batches,
                        "loss": loss.item(),
                    },
                    ["epoch", "batch", "loss"],
                )
            if num_batches == 1 or num_batches % 5 == 0:
                print(
                    f"  batch {num_batches}/{len(train_loader)} — loss: {loss.item():.4f}",
                    flush=True,
                )

        print(f"Epoch {epoch + 1}/{num_epochs} — loss: {total_loss / num_batches:.4f}", flush=True)


def evaluate_reward_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            chosen_ids    = batch["chosen_ids"].to(device)
            chosen_mask   = batch["chosen_mask"].to(device)
            rejected_ids  = batch["rejected_ids"].to(device)
            rejected_mask = batch["rejected_mask"].to(device)

            reward_chosen   = model(chosen_ids, chosen_mask)
            reward_rejected = model(rejected_ids, rejected_mask)

            correct += (reward_chosen > reward_rejected).sum().item()
            total   += reward_chosen.size(0)

    return correct / total
