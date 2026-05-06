import torch
import torch.nn as nn
from transformers import AutoModel

class RewardModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        self.reward_head = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask = attention_mask)

        last_idx = attention_mask.sum(dim = 1) - 1
        batch_idx = torch.arange(input_ids.size(0), device = input_ids.device)


        last_hidden = outputs.last_hidden_state[batch_idx, last_idx, :]
        score = self.reward_head(last_hidden)
        return score
