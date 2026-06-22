"""
9： —— PPO vs TD3 vs SAC
——

：
    python ppo_td3_sac_comparison.py

：

    PPO（Proximal Policy Optimization）—— ""
        ：（on-policy）
        ：（）
        ：，
        ：、、
        ：（）
        ：、

    TD3（Twin Delayed DDPG）—— ""
        ：（off-policy）
        ：（）
        ： Q  +  + 
        ：、
        ：、
        ：、

    SAC（Soft Actor-Critic）—— ""
        ：（off-policy）
        ：（ + ）
        ： +  +  Q 
        ：、、
        ：、
        ：、

：
    - （HalfCheetah-v4）
    - （50000 ）
    - 
    - 
"""

import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, TD3, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

# 
os.makedirs("output", exist_ok=True)

# 
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# ： —— 
# ==========================================
class RewardCallback(BaseCallback):
    """
    ：

     SB3 ，
     info["episode"]["r"] 。
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_timesteps = []  # 

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_timesteps.append(self.num_timesteps)
        return True


# ==========================================
# ：
# ==========================================
print("=" * 60)
print("9： — PPO vs TD3 vs SAC")
print("=" * 60)

#  HalfCheetah-v4（ MuJoCo）
#  MuJoCo ， Pendulum-v1
ENV_NAME = "HalfCheetah-v4"
try:
    test_env = gym.make(ENV_NAME)
    test_env.reset()
    test_env.close()
    print(f"\n: {ENV_NAME}（MuJoCo ）")
except Exception as e:
    ENV_NAME = "Pendulum-v1"
    print(f"\nMuJoCo （{e}），: {ENV_NAME}")

# 
env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_type = "" if isinstance(env.action_space, gym.spaces.Box) else ""
print(f"  :   {state_dim}")
print(f"  :   {action_dim}")
print(f"  :   {action_type}")
env.close()

# 
TOTAL_TIMESTEPS = 50_000    # （， 1M+）
SEED = 42                   # 
NET_ARCH = [256, 256]       # 


# ==========================================
# ：
# ==========================================

# ----  1：PPO ----
print("\n" + "-" * 60)
print("【1/3】 PPO（Proximal Policy Optimization）")
print("-" * 60)
print("  ：、、")

# PPO ：
#   - n_steps=2048: ，PPO ""
#   - batch_size=64: 
#   - n_epochs=10:  10 
#   - clip_range=0.2: 
#   - ent_coef=0.01: （， SAC ）
ppo_model = PPO(
    policy="MlpPolicy",
    env=ENV_NAME,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    clip_range=0.2,
    ent_coef=0.01,
    gamma=0.99,
    gae_lambda=0.95,
    verbose=0,
    seed=SEED,
    device="auto",
    policy_kwargs=dict(net_arch=NET_ARCH),
)

ppo_callback = RewardCallback()
ppo_model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=ppo_callback,
    progress_bar=True,
)
print(f"  PPO ， {len(ppo_callback.episode_rewards)} ")


# ----  2：TD3 ----
print("\n" + "-" * 60)
print("【2/3】 TD3（Twin Delayed DDPG）")
print("-" * 60)
print("  ：、、Q、")

# TD3 （ DDPG ）：
#   1.  Q （Clipped Double-Q）： Q 
#      →  Q 
#   2. （Delayed Policy Updates）：
#      → Critic ，Actor 
#      → policy_delay=2  2  Critic ， 1  Actor 
#   3. （Target Policy Smoothing）：
#      → ， Q 
td3_model = TD3(
    policy="MlpPolicy",
    env=ENV_NAME,
    learning_rate=3e-4,
    buffer_size=100_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    policy_delay=2,           # ： 2  Critic  Actor
    action_noise=None,        # （TD3 ）
    verbose=0,
    seed=SEED,
    device="auto",
    policy_kwargs=dict(net_arch=NET_ARCH),
)

td3_callback = RewardCallback()
td3_model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=td3_callback,
    progress_bar=True,
)
print(f"  TD3 ， {len(td3_callback.episode_rewards)} ")


# ----  3：SAC ----
print("\n" + "-" * 60)
print("【3/3】 SAC（Soft Actor-Critic）")
print("-" * 60)
print("  ：、、、")

# SAC  —— ：
#   ：max Σ r(s,a)
#   ：max Σ [r(s,a) + α * H(π(·|s))]
#
#  H ，α 
#  SAC ，
#
# ：
#   alpha 
#    = -dim(A)（）
#   （）→ alpha  → 
#   （）→ alpha  → 
sac_model = SAC(
    policy="MlpPolicy",
    env=ENV_NAME,
    learning_rate=3e-4,
    buffer_size=100_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    ent_coef="auto",          # （SAC ！）
    train_freq=1,
    gradient_steps=1,
    verbose=0,
    seed=SEED,
    device="auto",
    policy_kwargs=dict(net_arch=NET_ARCH),
)

sac_callback = RewardCallback()
sac_model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=sac_callback,
    progress_bar=True,
)
print(f"  SAC ， {len(sac_callback.episode_rewards)} ")


# ==========================================
# ：
# ==========================================
print("\n" + "=" * 60)
print("： 10 ")
print("=" * 60)

eval_env = gym.make(ENV_NAME)
n_eval = 10

results = {}
for name, model in [("PPO", ppo_model), ("TD3", td3_model), ("SAC", sac_model)]:
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=n_eval, deterministic=True
    )
    # 
    test_rewards = []
    for _ in range(n_eval):
        obs, _ = eval_env.reset()
        done, truncated = False, False
        total_r = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, truncated, _ = eval_env.step(action)
            total_r += r
        test_rewards.append(total_r)

    results[name] = {
        "mean": mean_reward,
        "std": std_reward,
        "rewards": test_rewards,
    }
    print(f"  {name:4s}:  = {mean_reward:8.2f} ± {std_reward:6.2f}")

eval_env.close()


# ==========================================
# ：
# ==========================================
print("\n...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    f" — {ENV_NAME}（{TOTAL_TIMESTEPS:,} ）",
    fontsize=16, fontweight="bold",
)

# 
colors = {"PPO": "#2196F3", "TD3": "#F44336", "SAC": "#4CAF50"}

# 1：（）
ax1 = axes[0, 0]
for name, cb in [("PPO", ppo_callback), ("TD3", td3_callback), ("SAC", sac_callback)]:
    if cb.episode_rewards:
        ax1.plot(cb.episode_rewards, alpha=0.3, color=colors[name], linewidth=0.8)
ax1.set_title("（）", fontsize=13)
ax1.set_xlabel("")
ax1.set_ylabel("")
# 
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=colors[n], linewidth=2, label=n)
    for n in ["PPO", "TD3", "SAC"]
]
ax1.legend(handles=legend_elements)
ax1.grid(True, alpha=0.3)

# 2：（）
ax2 = axes[0, 1]
for name, cb in [("PPO", ppo_callback), ("TD3", td3_callback), ("SAC", sac_callback)]:
    if cb.episode_rewards:
        rewards = cb.episode_rewards
        window = min(20, len(rewards))
        if window > 1:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax2.plot(range(window - 1, len(rewards)), smoothed,
                     color=colors[name], linewidth=2, label=f"{name}")
ax2.set_title("（）", fontsize=13)
ax2.set_xlabel("")
ax2.set_ylabel("")
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3：（）
ax3 = axes[1, 0]
algo_names = list(results.keys())
means = [results[n]["mean"] for n in algo_names]
stds = [results[n]["std"] for n in algo_names]
bar_colors = [colors[n] for n in algo_names]
bars = ax3.bar(algo_names, means, yerr=stds, color=bar_colors,
               alpha=0.8, capsize=5, edgecolor="white", linewidth=1.5)
ax3.set_title("（10 ）", fontsize=13)
ax3.set_ylabel("")
# 
for bar, mean, std in zip(bars, means, stds):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 10,
             f"{mean:.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax3.grid(True, alpha=0.3, axis="y")

# 4：（）
ax4 = axes[1, 1]
box_data = [results[n]["rewards"] for n in algo_names]
bp = ax4.boxplot(box_data, labels=algo_names, patch_artist=True, widths=0.5)
for patch, color in zip(bp["boxes"], bar_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax4.set_title("", fontsize=13)
ax4.set_ylabel("")
ax4.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("output/ppo_td3_sac_comparison.png", dpi=150, bbox_inches="tight")
print(": output/ppo_td3_sac_comparison.png")
plt.show()


# ==========================================
# ：
# ==========================================
print("\n" + "=" * 60)
print("")
print("=" * 60)

# 
print(f"{'':<20s} {'PPO':>10s} {'TD3':>10s} {'SAC':>10s}")
print("-" * 60)

# 
print(f"{'':<18s}", end="")
for name in algo_names:
    print(f" {results[name]['mean']:>10.1f}", end="")
print()

# 
print(f"{'':<18s}", end="")
for name in algo_names:
    print(f" {results[name]['std']:>10.1f}", end="")
print()

# 
print(f"{'':<18s}", end="")
for cb in [ppo_callback, td3_callback, sac_callback]:
    print(f" {len(cb.episode_rewards):>10d}", end="")
print()

# 
print(f"{'':<18s} {'':>10s} {'':>10s} {'':>10s}")

# 
print(f"{'':<18s} {'':>10s} {'':>10s} {'+':>10s}")

# 
print(f"{'':<18s} {'':>10s} {'':>10s} {'':>10s}")

# 
print(f"{'':<18s} {'':>10s} {'':>10s} {'':>10s}")

# 
print(f"{'':<18s} {'':>10s} {'':>10s} {'':>10s}")

print("-" * 60)

# 
winner = max(results.keys(), key=lambda k: results[k]["mean"])
print(f"\n，{winner} ！")
print()
print("：")
print("  - 50k ， 1M+ ")
print("  - ")
print("  - PPO ")
print("  - SAC  MuJoCo （）")
print("  - TD3 （）")
print("=" * 60)
