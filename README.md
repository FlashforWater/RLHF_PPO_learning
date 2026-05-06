# RLHF PPO Learning Project

This repository is a small, readable implementation of the two-stage RLHF pipeline:

1. Train a reward model on chosen/rejected preference pairs.
2. Use PPO to fine-tune a language model against that reward model while penalizing KL drift from a frozen reference model.

The code is intentionally direct rather than production-optimized, so it is suitable for learning the mechanics of reward modeling, rollout generation, KL shaping, GAE, and PPO loss.

## Project Layout

```text
.
├── config.py                 # Model and training hyperparameters
├── train_reward.py           # Stage 1: train reward_model.pt
├── train_ppo.py              # Stage 2: train ppo_policy.pt and ppo_value.pt
├── eval_reward.py            # Reward-model accuracy and reward-gap evaluation
├── inference.py              # Compare base model and PPO policy outputs
├── explore_data.py           # Small dataset/tokenizer exploration script
├── data/
│   └── data_loader.py        # Anthropic HH-RLHF dataset loaders
├── models/
│   ├── reward_model.py       # LM backbone + scalar reward head
│   └── value_model.py        # LM backbone + per-token value head
├── rlhf/
│   ├── reward_trainer.py     # Bradley-Terry reward-model training
│   ├── rollout.py            # Sampling and log-prob collection
│   ├── kl_penalty.py         # Per-token KL reward shaping
│   ├── advantage.py          # GAE advantage/return calculation
│   ├── ppo_loss.py           # PPO clipped policy and value losses
│   └── ppo_trainer.py        # One PPO train step
└── tests/
    ├── test_advantage.py
    └── test_data_loader.py
```

## Setup

Use a fresh virtual environment on your local machine or server.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you use the default model in `config.py`, you may need Hugging Face access to `meta-llama/Llama-3.2-1B-Instruct`:

```bash
huggingface-cli login
```

For a smaller first pass, change `ModelConfig.model_name` in `config.py` to a smaller open model that fits your hardware.

## Verify

Run the local unit tests:

```bash
python -m unittest discover -s tests
```

Run a syntax check:

```bash
PYTHONPYCACHEPREFIX=.pycache_check python -m py_compile \
  config.py train_reward.py train_ppo.py explore_data.py \
  data/data_loader.py models/reward_model.py models/value_model.py \
  rlhf/reward_trainer.py rlhf/rollout.py rlhf/ppo_trainer.py \
  rlhf/advantage.py rlhf/ppo_loss.py rlhf/kl_penalty.py
```

## Training Flow

Stage 1 trains the reward model:

```bash
python train_reward.py
```

This downloads Anthropic HH-RLHF through `datasets`, trains the scalar reward model, evaluates chosen-vs-rejected accuracy, and writes:

```text
reward_model.pt
runs/<timestamp>_reward/reward_model.pt
runs/<timestamp>_reward/reward_metrics.csv
```

Stage 2 runs PPO:

```bash
python train_ppo.py
```

This loads:

- trainable policy model
- frozen reference model
- frozen reward model from `reward_model.pt`
- trainable value model

It writes:

```text
ppo_policy.pt
ppo_value.pt
runs/<timestamp>_ppo/ppo_policy.pt
runs/<timestamp>_ppo/ppo_value.pt
runs/<timestamp>_ppo/ppo_metrics.csv
```

Evaluate the reward model:

```bash
python eval_reward.py
```

Compare base-model and PPO-policy generations:

```bash
python inference.py --prompt "How can I stay productive while studying?"
```

## PPO Step Map

One PPO iteration in `rlhf/ppo_trainer.py` does this:

1. Generate responses from the current policy.
2. Store old per-token log probabilities.
3. Score full prompt-response sequences with the reward model.
4. Compute reference-model log probabilities.
5. Build rewards from KL penalty plus terminal reward-model score.
6. Estimate values and compute masked GAE.
7. Recompute current policy log probabilities.
8. Optimize clipped PPO policy loss plus value loss for `ppo_epochs`.

## Server Deployment Notes

The repository is set up so you should push source code and config, not local training artifacts or the local virtual environment. `.gitignore` excludes:

- `venv/`
- Python caches
- model checkpoints such as `reward_model.pt`, `ppo_policy.pt`, and `ppo_value.pt`
- experiment output folders

Typical server flow:

```bash
git init
git add .
git commit -m "Initial RLHF PPO learning project"
git remote add origin <your-server-git-url>
git push -u origin main
```

If your server is accessed by SSH instead of Git, use:

```bash
rsync -av --exclude venv --exclude __pycache__ --exclude '*.pt' ./ <user>@<host>:/path/to/RLHF_PPO/
```

Then SSH into the server, create a fresh virtual environment, install requirements, log in to Hugging Face if needed, and run the training commands above.

## Hardware Notes

The default config loads multiple copies of the model during PPO. That is expensive:

- policy model
- reference model
- reward model backbone
- value model backbone

For learning, start with smaller batch sizes and a smaller model. Once the code path is clear, scale up gradually.
