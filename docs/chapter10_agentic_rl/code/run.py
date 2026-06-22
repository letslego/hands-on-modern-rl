# run.py
from transformers import AutoModelForCausalLM, AutoTokenizer

from environment import SandboxEnv
from policy import Policy
from trainer import GRPOAgentTrainer

# 
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 
env = SandboxEnv(timeout=10)
policy = Policy(model, tokenizer, lr=5e-5)
ref_model = AutoModelForCausalLM.from_pretrained(model_name)
policy.set_ref_model(ref_model)


#  reward：
def code_reward(trajectory):
    """，reward = 1， = 0。"""
    for interaction in trajectory["interactions"]:
        obs = interaction.get("observation", "")
        if obs and "ERROR" not in obs and "TIMEOUT" not in obs:
            return 1.0
    return 0.0


#  prompts
prompts = [
    " Python  10 。",
    "。",
    "。",
]

# 
trainer = GRPOAgentTrainer(
    policy=policy,
    env=env,
    reward_fn=code_reward,
    group_size=4,
    max_turns=3,
)
history = trainer.fit(prompts, n_steps=30)
