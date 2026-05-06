# RLHF PPO Learning Guide

## 1. What This Project Is

This project is a small RLHF-PPO training pipeline.

It teaches the core workflow used in alignment training:

```text
preference dataset
-> reward model training
-> policy rollout
-> reward scoring
-> KL penalty against reference model
-> advantage calculation
-> PPO policy/value update
```

The goal is not to train a powerful model first. The goal is to understand the full engineering and ML loop.

## 2. Current Project State

The current local config is a smoke-test setup:

```python
model_name = "sshleifer/tiny-gpt2"
max_train_examples = 2
max_eval_examples = 2
max_prompt_examples = 2
num_iterations = 5
```

This is correct for proving:

```text
Docker works
data loads
reward training runs
PPO runs
checkpoints save
```

For real server learning, later switch to:

```python
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
```

Then increase examples gradually.

## 3. Important Files

Read the project in this order:

```text
data/data_loader.py       -> how HH-RLHF data becomes tensors
models/reward_model.py    -> LM backbone + scalar reward head
rlhf/reward_trainer.py    -> Bradley-Terry preference loss
train_reward.py           -> reward model training entrypoint
eval_reward.py            -> reward-model accuracy and reward-gap check

rlhf/rollout.py           -> generate responses and collect log probs
rlhf/kl_penalty.py        -> penalize policy drift from reference model
rlhf/advantage.py         -> compute GAE advantages and returns
rlhf/ppo_loss.py          -> clipped PPO loss and value loss
rlhf/ppo_trainer.py       -> one PPO training step
train_ppo.py              -> PPO training entrypoint
inference.py              -> compare base model and PPO policy outputs
```

For each file, ask:

```text
What tensors enter?
What are their shapes?
What do they mean mathematically?
What tensors leave?
```

## 4. Reward Model Stage

The reward model learns:

```text
score(prompt + chosen_response) > score(prompt + rejected_response)
```

Loss:

```text
L = -log sigmoid(r_chosen - r_rejected)
```

Important metrics:

```text
training loss
validation accuracy
```

Good signs:

```text
loss decreases
validation accuracy > 0.50
eventually 0.60-0.75 on a useful setup
```

Bad signs:

```text
accuracy near 0.50 -> random reward model
loss NaN -> learning rate too high or numerical issue
training too slow -> model/data too large for hardware
```

## 5. PPO Stage

PPO uses four models:

```text
policy_model     -> trainable model generating responses
ref_model        -> frozen original model for KL control
reward_model     -> frozen model scoring responses
value_model      -> trainable model predicting future reward
```

One PPO step:

```text
1. policy generates response
2. store old token log probabilities
3. reward model scores full response
4. reference model computes reference log probabilities
5. reward = terminal reward - beta * KL
6. value model predicts values
7. GAE computes advantages and returns
8. PPO updates policy and value model
```

## 6. Key PPO Metrics

Current printed metrics:

```text
reward       -> reward model score
kl           -> policy drift from reference model
policy_loss  -> PPO policy optimization signal
value_loss   -> value prediction error
```

How to read them:

```text
reward rising slowly is good
KL too high means policy is drifting too much
value_loss exploding means value model is unstable
policy_loss can be small or negative; that is normal
```

Later, add more logging:
The project saves PPO metrics to:

```text
runs/<timestamp>_ppo/ppo_metrics.csv
```

Useful future metrics:

```text
response_len
entropy
clip_fraction
approx_kl
advantage_mean
advantage_std
```

## 7. Parameter Tuning Guide

Reward model parameters:

```python
lr
batch_size
num_epochs
max_length
max_train_examples
max_eval_examples
```

Rules:

```text
OOM -> reduce batch_size or max_length
too slow -> reduce max_train_examples
accuracy stuck near 0.50 -> increase examples or epochs
loss unstable -> reduce lr
```

PPO parameters:

```python
policy_lr
value_lr
batch_size
num_iterations
ppo_epochs
max_gen_length
beta
epsilon
gamma
lam
value_coeff
```

Most important ones:

```text
beta:
  KL penalty strength.
  higher beta = stay closer to reference model
  lower beta = more policy drift

policy_lr:
  policy update size.
  too high = unstable
  too low = no learning

max_gen_length:
  response length.
  longer = slower and more memory

ppo_epochs:
  how many times to reuse rollout batch.
  start with 1, later try 4

epsilon:
  PPO clipping range.
  standard value is 0.2
```

## 8. Recommended Experiment Path

### Experiment 1: Docker Smoke Test

```python
model_name = "sshleifer/tiny-gpt2"
max_train_examples = 2
max_eval_examples = 2
num_iterations = 5
```

Goal:

```text
prove the full pipeline works
```

### Experiment 2: Small Real Model

```python
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
max_train_examples = 100
max_eval_examples = 50
max_prompt_examples = 100
num_iterations = 20
max_gen_length = 64
```

Goal:

```text
understand runtime, memory, and logs
```

### Experiment 3: Better Reward Model

```python
max_train_examples = 1000
max_eval_examples = 200
num_epochs = 1
```

Goal:

```text
validation accuracy clearly above random
```

### Experiment 4: KL Tuning

Run three PPO jobs:

```python
beta = 0.005
beta = 0.02
beta = 0.05
```

Compare:

```text
reward
KL
response quality
```

## 9. Real Project Workflow

Use this professional workflow:

```text
GitHub = source code storage
Docker = reproducible environment
tmux = keep server training alive
logs = inspect progress
checkpoints = model outputs
```

Local machine:

```bash
git add .
git commit -m "Describe change"
git push
```

Server:

```bash
git pull
docker build -t rlhf-ppo .
tmux new -s rlhf
```

Inside tmux:

```bash
docker run --rm -it --gpus all \
  -v $(pwd):/app \
  -v $(pwd)/.cache:/root/.cache \
  rlhf-ppo \
  bash -lc "python train_reward.py 2>&1 | tee logs/train_reward.log"
```

Detach:

```text
Ctrl-b d
```

Reattach:

```bash
tmux attach -t rlhf
```

## 10. What You Should Learn

By the end, you should understand:

```text
how preference data becomes model supervision
why reward models use chosen/rejected pairs
why PPO needs old_log_probs
why KL penalty prevents model drift
why GAE converts rewards into advantages
why value models are trained alongside policy models
how Docker/Git/tmux make training reproducible
how to tune parameters based on logs
```

The core mental model:

```text
Reward model: "How good is this answer?"
Reference model: "How far did we move?"
Value model: "How much future reward should I expect?"
Policy model: "What token should I generate next?"
PPO: "Improve reward, but do not update too aggressively."
```
