"""
6： Stable-Baselines3  PPO  LunarLander-v3
—— PPO 

：
    python ppo_lunar_lander.py

PPO（）：
    1. （clip），""
    2. （epoch），
    3. （Actor-Critic ）
"""

import os
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


# ==========================================
# ： —— 
# ==========================================
class TrainingMonitorCallback(BaseCallback):
    """
    ： rollout  PPO 
    ：、、、 KL 
    """

    def __init__(self, check_freq=2048, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        # 
        self.episode_rewards = []
        self.entropy_list = []
        self.clip_fraction_list = []
        self.approx_kl_list = []
        self.timesteps_list = []

    def _on_step(self):
        # （）
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])

        #  rollout 
        if self.num_timesteps % self.check_freq == 0 and self.num_timesteps > 0:
            #  PPO 
            # entropy: ，
            # clip_fraction: ，
            # approx_kl:  KL ，
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

        return True


# ==========================================
# ：
# ==========================================
print("=" * 50)
print("6：PPO  LunarLander-v3")
print("=" * 50)

print("\n（4 ）...")

#  DummyVecEnv  4 
# ，
def make_env():
    """，"""
    def _init():
        env = gym.make("LunarLander-v3")
        return env
    return _init

num_envs = 4
vec_env = DummyVecEnv([make_env() for _ in range(num_envs)])
print(f" {num_envs} ")


# ==========================================
# ： PPO 
# ==========================================
print("\n PPO ...")

model = PPO(
    policy="MlpPolicy",       # 
    env=vec_env,              # 
    learning_rate=3e-4,       # ：Adam 
    n_steps=2048,             #  rollout （）
    batch_size=64,            # ：
    n_epochs=10,              # 
    clip_range=0.2,           # PPO ： [0.8, 1.2] 
    ent_coef=0.01,            # ：
    vf_coef=0.5,              # 
    gamma=0.99,               # 
    gae_lambda=0.95,          # GAE lambda：-
    verbose=1,
    seed=42,
    device="auto",
)

print(f"  :       {model.learning_rate}")
print(f"  Rollout : {model.n_steps}")
print(f"  :     {model.batch_size}")
print(f"  :     {model.n_epochs}")
print(f"  :     [{1 - model.clip_range:.1f}, {1 + model.clip_range:.1f}]")
print(f"  :       {model.ent_coef}")
print(f"  :     {model.vf_coef}")


# ==========================================
# ：
# ==========================================
print("\n（200000 ）...")
print("-" * 50)

# 
callback = TrainingMonitorCallback(check_freq=2048)

#  200,000 
total_timesteps = 200_000
model.learn(
    total_timesteps=total_timesteps,
    callback=callback,
    progress_bar=True,
)

print("-" * 50)
print("！")


# ==========================================
# ：
# ==========================================
print("\n...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("PPO  LunarLander-v3 — ", fontsize=16, fontweight="bold")

# 1：
ax1 = axes[0, 0]
if callback.episode_rewards:
    # 
    rewards = callback.episode_rewards
    window = min(20, len(rewards))
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    ax1.plot(smoothed, color="#2196F3", alpha=0.8, linewidth=1.5)
    ax1.set_title("（）", fontsize=13)
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.grid(True, alpha=0.3)

# 2：
ax2 = axes[0, 1]
if callback.entropy_list:
    ax2.plot(callback.timesteps_list, callback.entropy_list,
             color="#FF9800", alpha=0.8, linewidth=1.5)
    ax2.set_title("（）", fontsize=13)
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.grid(True, alpha=0.3)
    # ： = 

# 3：
ax3 = axes[1, 0]
if callback.clip_fraction_list:
    ax3.plot(callback.timesteps_list, callback.clip_fraction_list,
             color="#F44336", alpha=0.8, linewidth=1.5)
    ax3.axhline(y=0.2, color="gray", linestyle="--", alpha=0.5, label="clip_range=0.2")
    ax3.set_title("（clip fraction）", fontsize=13)
    ax3.set_xlabel("")
    ax3.set_ylabel("")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

# 4： KL 
ax4 = axes[1, 1]
if callback.approx_kl_list:
    ax4.plot(callback.timesteps_list, callback.approx_kl_list,
             color="#4CAF50", alpha=0.8, linewidth=1.5)
    ax4.set_title(" KL （）", fontsize=13)
    ax4.set_xlabel("")
    ax4.set_ylabel("KL ")
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("output/ppo_lunar_lander_curves.png", dpi=150, bbox_inches="tight")
print(": output/ppo_lunar_lander_curves.png")
plt.show()


# ==========================================
# ：
# ==========================================
print("\n（20 ）...")
print("-" * 50)

# 
eval_env = gym.make("LunarLander-v3")
mean_reward, std_reward = evaluate_policy(
    model, eval_env, n_eval_episodes=20, deterministic=True
)
print(f"20 ：")
print(f"  : {mean_reward:.2f}")
print(f"  :   {std_reward:.2f}")

# ，
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
    status = "" if r >= 200 else ""
    print(f"   {i + 1:2d}: {r:8.2f}  [{status}]")

print(f"\n（>= 200 ）: {sum(1 for r in test_rewards if r >= 200)}/20")
eval_env.close()


# ==========================================
# ：
# ==========================================
model.save("output/ppo_lunar_lander")
print(f"\n: output/ppo_lunar_lander.zip")
print("=" * 50)
