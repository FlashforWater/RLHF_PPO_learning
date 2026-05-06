from datasets import load_dataset
from transformers import AutoTokenizer

ds = load_dataset("Anthropic/hh-rlhf", split="train")


def split_prompt_response(text):
    pos = text.rfind("\n\nAssistant:") 
    prompt = text[:pos]
    response = text[pos + 13:]
    return prompt, response



def process_dataset(ds):
    result = []
    for i in range(len(ds)):
        chosen_prompt, chosen_response = split_prompt_response(ds[i]['chosen'])
        rejected_prompt, rejected_response = split_prompt_response(ds[i]['rejected'])
        result.append((chosen_prompt, chosen_response, rejected_response))
    return result

# test tokenizer
#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

def tokenize_pair(prompt, chosen_response, rejected_response, tokenizer):
    chosen_ids  = tokenizer(prompt + "\n\nAssistant: " + chosen_response, padding = "max_length", max_length = 512, truncation = True)
    rejected_ids = tokenizer(prompt + "\n\nAssistant: " + rejected_response, padding = "max_length", max_length = 512, truncation = True)
    return chosen_ids, rejected_ids

data = process_dataset(ds.select(range(1)))
chosen_ids, rejected_ids = tokenize_pair(data[0][0], data[0][1], data[0][2], tokenizer)
print("CHOSEN length:", len(chosen_ids['input_ids']))
print("REJECTED length:", len(rejected_ids['input_ids']))