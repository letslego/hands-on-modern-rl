"""
9： SAC（Soft Actor-Critic） HalfCheetah-v4
——

：
    python sac_halfcheetah.py

SAC ：
    1. （Entropy Regularization）：，
       → ，
    2. （Automatic Temperature Tuning）：alpha 
       → -
    3.  Q （Twin Critics）： Q ，
       →  TD3 ，

SAC ：
    J(π) = Σ_t E_{(s,a)~ρ_π}[r(s,a) + α * H(π(·|s))]
     H ，α 

 PPO、TD3 ：
    - PPO：（on-policy），
    - TD3：（off-policy），， Q 
    - SAC：（off-policy），，，
"""

import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

# 
os.makedirs("output", exist_ok=True)

# 
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# ： ——  SAC 
# ==========================================
class SACTrainingCallback(BaseCallback):
    """
    ： SAC 

    SAC ：
        - episode_reward：，
        - entropy/alpha：（），
        - critic_loss：Critic ，
        - actor_loss：Actor ，

     PPO ：
        - SAC  off-policy，， clip_fraction
        - SAC  alpha（）
        - SAC  Critic ， critic_loss
    """

    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        # 
        self.episode_rewards = []
        self.alpha_list = []          # （）
        self.critic_loss_list = []    # Critic 
        self.actor_loss_list = []     # Actor 
        self.entropy_list = []        # 
        self.timesteps_list = []      # 

    def _on_step(self):
        # （）
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])

        #  check_freq 
        if self.num_timesteps % self.check_freq == 0 and self.num_timesteps > 0:
            logger = self.model.logger
            if hasattr(logger, "name_to_value"):
                name_to_value = logger.name_to_value

                # alpha：SAC 
                # ，alpha 
                # alpha  → 
                # alpha  → 
                alpha = name_to_value.get("train/entropy_coef", 0)
                # critic_loss： Q 
                #  Q 
                critic_loss = name_to_value.get("train/critic_loss", 0)
                # actor_loss：
                #  Q 
                actor_loss = name_to_value.get("train/actor_loss", 0)
                # entropy：
                entropy = name_to_value.get("train/entropy", 0)

                self.alpha_list.append(alpha)
                self.critic_loss_list.append(critic_loss)
                self.actor_loss_list.append(actor_loss)
                self.entropy_list.append(entropy)
                self.timesteps_list.append(self.num_timesteps)

        return True


# ==========================================
# ：
# ==========================================
print("=" * 50)
print("9：SAC  HalfCheetah-v4（）")
print("=" * 50)

print("\n HalfCheetah-v4 ...")

# HalfCheetah  MuJoCo 
# ：
#   - ：17 （、）
#   - ：6 （）
#   - ：
#   - ： - 
env = gym.make("HalfCheetah-v4")

state_dim = env.observation_space.shape[0]   # 17
action_dim = env.action_space.shape[0]       # 6
action_low = env.action_space.low            # 
action_high = env.action_space.high          # 

print(f"  :   {state_dim}")
print(f"  :   {action_dim}")
print(f"  :   [{action_low[0]:.1f}, {action_high[0]:.1f}] × {action_dim}")
print(f"  :   （Box）")


# ==========================================
# ： SAC 
# ==========================================
print("\n SAC ...")

# SAC ：
#
# learning_rate=3e-4
#   。SAC  PPO 
#   ，
#
# buffer_size=100000
#   
#   SAC  off-policy ，
#   ，
#
# batch_size=256
#   
#   SAC  batch_size（256  512）
#    PPO  64 ， off-policy 
#
# tau=0.005
#   
#   θ_target ← τ * θ + (1 - τ) * θ_target
#    tau =  = ，
#
# gamma=0.99
#   ， PPO 
#   
#
# ent_coef="auto"
#   （SAC ！）
#   SAC  alpha 
#    = -dim(A) = -6（）

model = SAC(
    policy="MlpPolicy",          # 
    env=env,                     # 
    learning_rate=3e-4,          # 
    buffer_size=100_000,         # 
    batch_size=256,              # 
    tau=0.005,                   # 
    gamma=0.99,                  # 
    ent_coef="auto",             # ：（SAC ）
    target_update_interval=1,    # （）
    train_freq=1,                # （）
    gradient_steps=1,            # 
    verbose=1,
    seed=42,
    device="auto",
    policy_kwargs=dict(
        net_arch=[256, 256],     # ： PPO 
    ),
)

print(f"  :         {model.learning_rate}")
print(f"  :     {model.buffer_size}")
print(f"  :       {model.batch_size}")
print(f"   tau: {model.tau}")
print(f"   gamma: {model.gamma}")
print(f"  :     （ent_coef='auto'）")
print(f"  :         {-action_dim}（= -）")

#  SAC 
# SAC  Actor ： μ  σ
# ：a ~ tanh(N(μ, σ²))
#  tanh 
print(f"\n  : {model.policy}")


# ==========================================
# ：
# ==========================================
print("\n（100000 ）...")
print("-" * 50)

# 
callback = SACTrainingCallback(check_freq=1000)

#  100,000 （， 1M+）
total_timesteps = 100_000
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

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("SAC  HalfCheetah-v4 — ", fontsize=16, fontweight="bold")

# 1：
ax1 = axes[0, 0]
if callback.episode_rewards:
    rewards = callback.episode_rewards
    window = min(20, len(rewards))
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    ax1.plot(rewards, alpha=0.3, color="#90CAF9", label="")
    ax1.plot(range(window - 1, len(rewards)), smoothed,
             color="#2196F3", linewidth=2, label=f" (={window})")
ax1.set_title("", fontsize=13)
ax1.set_xlabel("")
ax1.set_ylabel("")
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2：（alpha）—— SAC 
ax2 = axes[0, 1]
if callback.alpha_list:
    ax2.plot(callback.timesteps_list, callback.alpha_list,
             color="#FF9800", alpha=0.8, linewidth=1.5)
    # ：alpha 
    ax2.annotate(
        "alpha \n→ ",
        xy=(callback.timesteps_list[-1] * 0.6,
            max(callback.alpha_list) * 0.7),
        fontsize=9, color="gray", style="italic",
    )
ax2.set_title(" alpha（）", fontsize=13)
ax2.set_xlabel("")
ax2.set_ylabel("alpha")
ax2.grid(True, alpha=0.3)

# 3：
ax3 = axes[0, 2]
if callback.entropy_list:
    ax3.plot(callback.timesteps_list, callback.entropy_list,
             color="#4CAF50", alpha=0.8, linewidth=1.5)
ax3.set_title("（）", fontsize=13)
ax3.set_xlabel("")
ax3.set_ylabel("")
ax3.grid(True, alpha=0.3)

# 4：Critic 
ax4 = axes[1, 0]
if callback.critic_loss_list:
    ax4.plot(callback.timesteps_list, callback.critic_loss_list,
             color="#F44336", alpha=0.8, linewidth=1.5)
ax4.set_title("Critic （ Q ）", fontsize=13)
ax4.set_xlabel("")
ax4.set_ylabel("")
ax4.grid(True, alpha=0.3)

# 5：Actor 
ax5 = axes[1, 1]
if callback.actor_loss_list:
    ax5.plot(callback.timesteps_list, callback.actor_loss_list,
             color="#9C27B0", alpha=0.8, linewidth=1.5)
ax5.set_title("Actor （）", fontsize=13)
ax5.set_xlabel("")
ax5.set_ylabel("")
ax5.grid(True, alpha=0.3)

# 6：
ax6 = axes[1, 2]
if callback.episode_rewards:
    # ，
    mid = len(callback.episode_rewards) // 2
    first_half = callback.episode_rewards[:mid]
    second_half = callback.episode_rewards[mid:]
    ax6.hist(first_half, bins=20, alpha=0.5, color="#90CAF9", label="")
    ax6.hist(second_half, bins=20, alpha=0.5, color="#2196F3", label="")
    ax6.legend()
ax6.set_title("（ vs ）", fontsize=13)
ax6.set_xlabel("")
ax6.set_ylabel("")
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("output/sac_halfcheetah_curves.png", dpi=150, bbox_inches="tight")
print(": output/sac_halfcheetah_curves.png")
plt.show()


# ==========================================
# ：
# ==========================================
print("\n（10 ）...")
print("-" * 50)

# 
eval_env = gym.make("HalfCheetah-v4")
mean_reward, std_reward = evaluate_policy(
    model, eval_env, n_eval_episodes=10, deterministic=True
)
print(f"10 ：")
print(f"  : {mean_reward:.2f}")
print(f"  :   {std_reward:.2f}")

# ，
test_rewards = []
for ep in range(10):
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
    print(f"   {i + 1:2d}: {r:8.2f}")

print(f"\n: {max(test_rewards):.2f}")
print(f": {min(test_rewards):.2f}")
print(f": {np.std(test_rewards):.2f}")
eval_env.close()


# ==========================================
# ：
# ==========================================
model.save("output/sac_halfcheetah")
print(f"\n: output/sac_halfcheetah.zip")

print("\n" + "=" * 50)
print("SAC ：")
print("  1. ：，")
print("  2. ：alpha ，")
print("  3.  Q ： Q ，")
print("  4. ：")
print("  5. ：，")
print("=" * 50)
