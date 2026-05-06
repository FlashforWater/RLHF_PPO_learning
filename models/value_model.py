import torch
import torch.nn as nn
from transformers import AutoModel

class ValueModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        self.value_head = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask = attention_mask)

        hidden = outputs.last_hidden_state
        values = self.value_head(hidden).squeeze(-1)
        return values
