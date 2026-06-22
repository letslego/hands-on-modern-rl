"""
7： Stable-Baselines3  PPO  BipedalWalker-v3
—— PPO 

：
    python ppo_bipedal_walker.py
    python ppo_bipedal_walker.py --total-timesteps 100000    # 
    python ppo_bipedal_walker.py --total-timesteps 2000000   # 

BipedalWalker-v3 ：
    1. （4 ）—— DQN ，PPO 
    2. 24 （10  +  + ）
    3.  LunarLander ，
    4. ""：100  >= 300
"""

import argparse
import os
from pathlib import Path
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# 
os.makedirs("output", exist_ok=True)

# 
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def parse_args():
    parser = argparse.ArgumentParser(description="PPO  BipedalWalker-v3")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
                        help="（ 1000000）")
    return parser.parse_args()


# ==========================================
# ： —— 
# ==========================================
class TrainingMonitorCallback(BaseCallback):
    """
    ： rollout  PPO 
    ：、、、 KL 
    
    """

    def __init__(self, check_freq=2048, checkpoint_steps=None, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.checkpoint_steps = checkpoint_steps or []
        self.episode_rewards = []
        self.entropy_list = []
        self.clip_fraction_list = []
        self.approx_kl_list = []
        self.timesteps_list = []
        self._saved_checkpoints = set()

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_info = info["episode"]
                self.episode_rewards.append(
                    ep_info["r"] if isinstance(ep_info, dict) else ep_info
                )

        if self.num_timesteps % self.check_freq == 0 and self.num_timesteps > 0:
            logger = self.model.logger
            if hasattr(logger, "name_to_value"):
                name_to_value = logger.name_to_value

                entropy = name_to_value.get("train/entropy_loss", 0)
                clip_frac = name_to_value.get("train/clip_fraction", 0)
                approx_kl = name_to_value.get("train/approx_kl", 0)

                self.entropy_list.append(entropy)
                self.clip_fraction_list.append(clip_frac)
                self.approx_kl_list.append(approx_kl)
                self.timesteps_list.append(self.num_timesteps)

        # 
        for ckpt_step in self.checkpoint_steps:
            if self.num_timesteps >= ckpt_step and ckpt_step not in self._saved_checkpoints:
                path = f"output/ppo_bipedal_walker_{ckpt_step // 1000}k"
                self.model.save(path)
                print(f"\n  []  {ckpt_step // 1000}k  → {path}.zip")
                self._saved_checkpoints.add(ckpt_step)

        return True


# ==========================================
# ：
# ==========================================
args = parse_args()

print("=" * 50)
print("7：PPO  BipedalWalker-v3")
print("=" * 50)

print("\n（8 ）...")

def make_env():
    """"""
    def _init():
        env = gym.make("BipedalWalker-v3")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return _init

num_envs = 8
vec_env = DummyVecEnv([make_env() for _ in range(num_envs)])
print(f" {num_envs} ")


# ==========================================
# ： PPO 
# ==========================================
print("\n PPO ...")

model = PPO(
    policy="MlpPolicy",       # 
    env=vec_env,              # 
    learning_rate=3e-4,       # 
    n_steps=2048,             #  rollout （）
    batch_size=256,           # （ LunarLander ，）
    n_epochs=10,              # 
    clip_range=0.2,           # PPO 
    ent_coef=0.005,           # （，）
    vf_coef=0.5,              # 
    gamma=0.99,               # 
    gae_lambda=0.95,          # GAE lambda
    verbose=1,
    seed=42,
    device="auto",
)

clip_val = model.clip_range(1.0) if callable(model.clip_range) else model.clip_range
print(f"  :       {model.learning_rate}")
print(f"  Rollout : {model.n_steps}")
print(f"  :     {model.batch_size}")
print(f"  :     {model.n_epochs}")
print(f"  :     [{1 - clip_val:.1f}, {1 + clip_val:.1f}]")
print(f"  :       {model.ent_coef}")
print(f"  :      {vec_env.num_envs} （）")


# ==========================================
# ：
# ==========================================
total_timesteps = args.total_timesteps
print(f"\n（{total_timesteps:,} ）...")
print("-" * 50)

# （）
checkpoint_steps = []
if total_timesteps >= 500_000:
    checkpoint_steps = [100_000, 500_000]
callback = TrainingMonitorCallback(check_freq=2048, checkpoint_steps=checkpoint_steps)

model.learn(
    total_timesteps=total_timesteps,
    callback=callback,
    progress_bar=True,
)

print("-" * 50)
print("！")


# ==========================================
# ：（4 ）
# ==========================================
print("\n...")

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# 1：（ + ）
if callback.episode_rewards:
    rewards = callback.episode_rewards
    window = min(50, max(1, len(rewards)))
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rewards, color="#90CAF9", alpha=0.4, linewidth=0.8, label="")
    x_smooth = np.arange(window // 2, window // 2 + len(smoothed))
    ax.plot(x_smooth, smoothed, color="#1565C0", alpha=0.9, linewidth=1.8, label="50 ")
    ax.axhline(y=300, color="green", linestyle="--", alpha=0.5, label="solved (300)")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
    ax.set_title("PPO BipedalWalker-v3 ", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "ppo_bipedal_walker_reward.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   → output/ppo_bipedal_walker_reward.png")

# 2：
if callback.entropy_list:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(callback.timesteps_list, callback.entropy_list,
            color="#FF9800", alpha=0.8, linewidth=1.5)
    ax.set_title("PPO BipedalWalker-v3 ", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "ppo_bipedal_walker_entropy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   → output/ppo_bipedal_walker_entropy.png")

# 3：
if callback.clip_fraction_list:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(callback.timesteps_list, callback.clip_fraction_list,
            color="#F44336", alpha=0.8, linewidth=1.5, label="clip fraction")
    ax.axhline(y=0.2, color="gray", linestyle="--", alpha=0.5, label="clip_range=0.2")
    ax.set_title("PPO BipedalWalker-v3 ", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "ppo_bipedal_walker_clip.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   → output/ppo_bipedal_walker_clip.png")

# 4： KL 
if callback.approx_kl_list:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(callback.timesteps_list, callback.approx_kl_list,
            color="#4CAF50", alpha=0.8, linewidth=1.5)
    ax.set_title("PPO BipedalWalker-v3  KL ", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("KL ")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "ppo_bipedal_walker_kl.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  KL  → output/ppo_bipedal_walker_kl.png")


# ==========================================
# ：
# ==========================================
print("\n（20 ）...")
print("-" * 50)

eval_env = gym.make("BipedalWalker-v3")
mean_reward, std_reward = evaluate_policy(
    model, eval_env, n_eval_episodes=20, deterministic=True
)
print(f"20 ：")
print(f"  : {mean_reward:.2f}")
print(f"  :   {std_reward:.2f}")

test_rewards = []
for ep in range(20):
    obs, _ = eval_env.reset()
    done, truncated = False, False
    total_reward = 0.0
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = eval_env.step(action)
        total_reward += reward
    test_rewards.append(total_reward)

print(f"\n：")
for i, r in enumerate(test_rewards):
    status = "" if r >= 300 else ("" if r >= 100 else "")
    print(f"   {i + 1:2d}: {r:8.2f}  [{status}]")

print(f"\n（>= 300 ）: {sum(1 for r in test_rewards if r >= 300)}/20")
eval_env.close()


# ==========================================
# ：
# ==========================================
model.save("output/ppo_bipedal_walker")
print(f"\n: output/ppo_bipedal_walker.zip")
print("=" * 50)
