from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Tiny model for local CPU/Docker smoke tests. Switch to
    # "Qwen/Qwen2.5-0.5B-Instruct" or a larger model on a real GPU server.
    model_name: str = 'sshleifer/tiny-gpt2'


@dataclass
class RewardTrainConfig:
    lr: float = 1e-5
    batch_size: int = 1
    num_epochs: int = 1
    max_length: int = 128
    max_train_examples: int = 2
    max_eval_examples: int = 2


@dataclass
class PPOConfig:
    policy_lr: float = 5e-6
    value_lr: float = 1e-5
    batch_size: int = 1
    num_iterations: int = 5
    ppo_epochs: int = 1
    max_gen_length: int = 32
    max_prompt_examples: int = 2
    beta: float = 0.02
    lam: float = 0.95
    gamma: float = 1.0
    epsilon: float = 0.2
    value_coeff: float = 0.5 
