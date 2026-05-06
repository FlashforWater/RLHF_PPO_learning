import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ModelConfig


def generate(model, tokenizer, prompt, device, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--policy-path", default="ppo_policy.pt")
    args = parser.parse_args()

    model_config = ModelConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        dtype=torch.float32,
    ).to(device)
    base_model.eval()

    print("Loading PPO policy...")
    ppo_model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        dtype=torch.float32,
    ).to(device)
    ppo_model.load_state_dict(torch.load(args.policy_path, map_location=device))
    ppo_model.eval()

    print("\nBASE MODEL:")
    print(generate(base_model, tokenizer, args.prompt, device, args.max_new_tokens))

    print("\nPPO MODEL:")
    print(generate(ppo_model, tokenizer, args.prompt, device, args.max_new_tokens))


if __name__ == "__main__":
    main()
