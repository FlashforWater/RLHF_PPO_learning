import torch
def generate_with_log_probs(policy_model, input_ids, attention_mask, max_gen_length, eos_token_id):
    batch_size = input_ids.size(0)
    device = input_ids.device

    log_probs_list = []
    response_mask_list = []
    has_ended = torch.zeros(batch_size, dtype = torch.bool, device = device)

    for _ in range(max_gen_length):
        outputs = policy_model(input_ids = input_ids, attention_mask = attention_mask)
        last_logits = outputs.logits[:, -1, :]

        probs = torch.softmax(last_logits, dim = -1)
        next_token = torch.multinomial(probs, num_samples = 1)
        token_prob = probs.gather(1, next_token)
        log_prob = torch.log(token_prob + 1e-10) # prevent log(0) from being -inf


        # "is this step alive" -- 1 if we haven't ended yet, 0 if already ended
        step_alive = (~has_ended).long().unsqueeze(-1)               # [B, 1]
        response_mask_list.append(step_alive)
        log_probs_list.append(log_prob)

        input_ids = torch.cat([input_ids, next_token], dim = 1)
        attention_mask = torch.cat([attention_mask, step_alive], dim = 1)
        has_ended = has_ended | (next_token.squeeze(-1) == eos_token_id)

    log_probs = torch.cat(log_probs_list, dim = 1)
    response_mask = torch.cat(response_mask_list, dim = 1).float()

    return input_ids, attention_mask, log_probs, response_mask


def compute_log_probs(model, full_sequence, attention_mask, response_length):
    prompt_len = full_sequence.shape[1] - response_length
    outputs = model(full_sequence, attention_mask = attention_mask)

    response_logits = outputs.logits[:, prompt_len -1 : -1, :]
    log_probs_all = torch.log_softmax(response_logits, dim = -1)

    response_tokens = full_sequence[:, prompt_len:]
    log_probs = log_probs_all.gather(2, response_tokens.unsqueeze(-1)).squeeze(-1)
    return log_probs




