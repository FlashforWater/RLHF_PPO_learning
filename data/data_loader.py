import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


ASSISTANT_SEP = "\n\nAssistant:"


def split_prompt_response(text):
    """
    Split a HH-RLHF conversation at its last "\\n\\nAssistant:" marker.

    Returns:
        prompt:   everything up to (and including) the last "\\n\\nHuman:" turn
        response: the assistant's final reply, with leading whitespace stripped
    """
    pos = text.rfind(ASSISTANT_SEP)
    if pos == -1:
        raise ValueError(f"Could not find assistant separator {ASSISTANT_SEP!r}")
    prompt = text[:pos]
    response = text[pos + len(ASSISTANT_SEP):].lstrip()
    return prompt, response


# ---------- Reward-model training: chosen / rejected pairs ----------

class HHPreferenceDataset(Dataset):
    def __init__(self, split, tokenizer, max_length=512, max_examples=None):
        ds = load_dataset("Anthropic/hh-rlhf", split=split)
        if max_examples is not None:
            ds = ds.select(range(min(max_examples, len(ds))))

        self.examples = []
        for item in ds:
            prompt, chosen_resp = split_prompt_response(item["chosen"])
            _,      rejected_resp = split_prompt_response(item["rejected"])
            self.examples.append((prompt, chosen_resp, rejected_resp))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        prompt, chosen_resp, rejected_resp = self.examples[idx]
        chosen_text   = prompt + ASSISTANT_SEP + " " + chosen_resp
        rejected_text = prompt + ASSISTANT_SEP + " " + rejected_resp

        enc_c = self.tokenizer(
            chosen_text, padding="max_length", max_length=self.max_length,
            truncation=True, return_tensors="pt",
        )
        enc_r = self.tokenizer(
            rejected_text, padding="max_length", max_length=self.max_length,
            truncation=True, return_tensors="pt",
        )
        return {
            "chosen_ids":    enc_c["input_ids"].squeeze(0),
            "chosen_mask":   enc_c["attention_mask"].squeeze(0),
            "rejected_ids":  enc_r["input_ids"].squeeze(0),
            "rejected_mask": enc_r["attention_mask"].squeeze(0),
        }


def get_reward_dataloader(split, tokenizer, batch_size, max_length=512, shuffle=True, max_examples=None):
    dataset = HHPreferenceDataset(
        split,
        tokenizer,
        max_length=max_length,
        max_examples=max_examples,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ---------- PPO training: prompts only ----------

class HHPromptDataset(Dataset):
    def __init__(self, split, tokenizer, max_length=256, max_examples=None):
        ds = load_dataset("Anthropic/hh-rlhf", split=split)
        if max_examples is not None:
            ds = ds.select(range(min(max_examples, len(ds))))

        self.prompts = []
        for item in ds:
            prompt, _ = split_prompt_response(item["chosen"])
            # add the "Assistant:" cue so the policy knows to continue as the assistant
            self.prompts.append(prompt + ASSISTANT_SEP)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.prompts[idx], padding="max_length", max_length=self.max_length,
            truncation=True, return_tensors="pt",
        )
        return {
            "prompt_ids":  enc["input_ids"].squeeze(0),
            "prompt_mask": enc["attention_mask"].squeeze(0),
        }


def get_ppo_dataloader(split, tokenizer, batch_size, max_length=256, shuffle=True, max_examples=None):
    dataset = HHPromptDataset(
        split,
        tokenizer,
        max_length=max_length,
        max_examples=max_examples,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
