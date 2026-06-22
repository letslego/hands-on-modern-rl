"""
6：GAE（）
—— λ  γ -

GAE ：
    δ_t = r_t + γ * V(s_{t+1}) - V(s_t)           # TD 
    A_t^GAE(γ,λ) = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}   # GAE 

λ ：
    λ → 0: 、（ TD ）
    λ → 1: 、（）

γ ：
    γ → 0: （）
    γ → 1: （）

：
    python gae_visualization.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# 
os.makedirs("output", exist_ok=True)

# 
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# ：GAE 
# ==========================================
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
     (GAE)

    ：
        rewards: 
        values:   V(s)
        dones:   
        gamma:   
        lam:     GAE lambda

    ：
        advantages: 
        returns:    
    """
    advantages = []
    gae = 0

    #  V(s_T+1)=0
    values = list(values) + [0.0]

    #  GAE
    for t in reversed(range(len(rewards))):
        if dones[t]:
            # ，
            gae = 0
            next_value = 0.0
        else:
            next_value = values[t + 1]

        # TD ：δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * next_value - values[t]

        # GAE ：A_t = δ_t + γλ * A_{t+1}
        gae = delta + gamma * lam * gae

        advantages.insert(0, gae)

    #  =  + 
    returns = [a + v for a, v in zip(advantages, values[:-1])]

    return advantages, returns


def compute_mc_returns(rewards, gamma=0.99):
    """
    （）
    
    """
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def compute_td_residuals(rewards, values, gamma=0.99):
    """
     TD 
    δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
    """
    values = list(values) + [0.0]
    residuals = []
    for t in range(len(rewards)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        residuals.append(delta)
    return residuals


# ==========================================
# ：
# ==========================================
print("=" * 60)
print("6：GAE（）")
print("=" * 60)

# ：5
# 4， +1
#  RL ""
rewards = [0.0, 0.0, 0.0, 0.0, 1.0]
n_steps = len(rewards)

# （）
# V(s) 
values = [0.1, 0.2, 0.4, 0.6, 0.9]

# 
dones = [False] * n_steps

print(f"\n:")
print(f"  :     {rewards}")
print(f"  :     {values}")
print(f"  :  — ")

# （）
mc_returns = compute_mc_returns(rewards, gamma=0.99)
print(f"  MC :      {[f'{r:.4f}' for r in mc_returns]}")

#  TD 
td_residuals = compute_td_residuals(rewards, values, gamma=0.99)
print(f"  TD :      {[f'{r:.4f}' for r in td_residuals]}")


# ==========================================
# ： λ  GAE 
# ==========================================
print("\n" + "=" * 60)
print(" λ  GAE ")
print("=" * 60)

lambda_values = [0.0, 0.5, 0.9, 0.95, 1.0]
gamma_fixed = 0.99

#  λ 
advantages_by_lambda = {}
returns_by_lambda = {}

for lam in lambda_values:
    adv, ret = compute_gae(rewards, values, dones, gamma=gamma_fixed, lam=lam)
    advantages_by_lambda[lam] = adv
    returns_by_lambda[lam] = ret

# 
print(f"\n{'λ ':<8}", end="")
for t in range(n_steps):
    print(f"{' ' + str(t):>12}", end="")
print()
print("-" * (8 + 12 * n_steps))

for lam in lambda_values:
    label = f"{lam:<8.2f}"
    print(label, end="")
    for t in range(n_steps):
        print(f"{advantages_by_lambda[lam][t]:>12.4f}", end="")
    print()

print(f"\n:")
print(f"  λ=0.0:  TD  → 、")
print(f"  λ=1.0:    → 、")
print(f"  λ=0.95: PPO   → ")


# ==========================================
# ： γ  GAE 
# ==========================================
print("\n" + "=" * 60)
print(" γ  GAE （ λ=0.95）")
print("=" * 60)

gamma_values = [0.5, 0.9, 0.95, 0.99, 1.0]
lambda_fixed = 0.95

advantages_by_gamma = {}
returns_by_gamma = {}

for gamma in gamma_values:
    adv, ret = compute_gae(rewards, values, dones, gamma=gamma, lam=lambda_fixed)
    advantages_by_gamma[gamma] = adv
    returns_by_gamma[gamma] = ret

# 
print(f"\n{'γ ':<8}", end="")
for t in range(n_steps):
    print(f"{' ' + str(t):>12}", end="")
print()
print("-" * (8 + 12 * n_steps))

for gamma in gamma_values:
    label = f"{gamma:<8.2f}"
    print(label, end="")
    for t in range(n_steps):
        print(f"{advantages_by_gamma[gamma][t]:>12.4f}", end="")
    print()

print(f"\n:")
print(f"  γ=0.5:   — ")
print(f"  γ=0.99: PPO  — ")
print(f"  γ=1.0:   — ")


# ==========================================
# ：
# ==========================================
print("\n...")

# ：22
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("GAE  — ", fontsize=18, fontweight="bold")

# 
colors_lambda = ["#F44336", "#FF9800", "#4CAF50", "#2196F3", "#9C27B0"]
colors_gamma = ["#E91E63", "#FF5722", "#009688", "#3F51B5", "#000000"]

steps = np.arange(n_steps)
step_labels = [f" {i}\n(r={rewards[i]})" for i in range(n_steps)]

# ---- 1： λ  ----
ax1 = axes[0, 0]
for i, lam in enumerate(lambda_values):
    adv = advantages_by_lambda[lam]
    ax1.plot(steps, adv, marker="o", linewidth=2.5, markersize=8,
             color=colors_lambda[i], label=f"λ = {lam}")

ax1.set_xticks(steps)
ax1.set_xticklabels(step_labels)
ax1.set_title(" λ ", fontsize=14, fontweight="bold")
ax1.set_ylabel(" A(s)", fontsize=12)
ax1.legend(fontsize=11, loc="upper left")
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

#  λ 
ax1.annotate("λ→0: 、\n（ TD）", xy=(0.5, 0.02),
             xycoords="axes fraction", fontsize=10, color="#F44336",
             style="italic", ha="left")
ax1.annotate("λ→1: 、\n（）", xy=(0.5, 0.15),
             xycoords="axes fraction", fontsize=10, color="#9C27B0",
             style="italic", ha="left")

# ---- 2： γ  ----
ax2 = axes[0, 1]
for i, gamma in enumerate(gamma_values):
    adv = advantages_by_gamma[gamma]
    ax2.plot(steps, adv, marker="s", linewidth=2.5, markersize=8,
             color=colors_gamma[i], label=f"γ = {gamma}")

ax2.set_xticks(steps)
ax2.set_xticklabels(step_labels)
ax2.set_title(" γ （λ=0.95）", fontsize=14, fontweight="bold")
ax2.set_ylabel(" A(s)", fontsize=12)
ax2.legend(fontsize=11, loc="upper left")
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

#  γ 
ax2.annotate("γ→0: \n（）", xy=(0.02, 0.02),
             xycoords="axes fraction", fontsize=10, color="#E91E63",
             style="italic", ha="left")
ax2.annotate("γ→1: \n（）", xy=(0.02, 0.15),
             xycoords="axes fraction", fontsize=10, color="#000000",
             style="italic", ha="left")

# ---- 3： λ  ----
ax3 = axes[1, 0]
for i, lam in enumerate(lambda_values):
    ret = returns_by_lambda[lam]
    ax3.plot(steps, ret, marker="o", linewidth=2.5, markersize=8,
             color=colors_lambda[i], label=f"λ = {lam}")

#  MC 
ax3.plot(steps, mc_returns, marker="*", linewidth=2, markersize=12,
         color="black", linestyle="--", label="MC  ()")

ax3.set_xticks(steps)
ax3.set_xticklabels(step_labels)
ax3.set_title(" λ ", fontsize=14, fontweight="bold")
ax3.set_xlabel("", fontsize=12)
ax3.set_ylabel(" G(s)", fontsize=12)
ax3.legend(fontsize=10, loc="upper left")
ax3.grid(True, alpha=0.3)

# ---- 4：- ----
ax4 = axes[1, 1]

# 
lams = np.linspace(0, 1, 100)
#  λ （）
bias = np.exp(-3 * lams) * 1.0
#  λ （）
variance = (np.exp(2 * lams) - 1) / (np.exp(2) - 1) * 1.0
#  = ² + 
total_error = bias ** 2 + variance

ax4.fill_between(lams, 0, bias ** 2, alpha=0.3, color="#2196F3", label="²")
ax4.fill_between(lams, bias ** 2, bias ** 2 + variance, alpha=0.3, color="#F44336", label="")
ax4.plot(lams, total_error, color="black", linewidth=2.5, label="")

#  λ 
optimal_idx = np.argmin(total_error)
optimal_lam = lams[optimal_idx]
ax4.axvline(x=optimal_lam, color="green", linestyle="--", linewidth=2, alpha=0.8)
ax4.annotate(f" λ ≈ {optimal_lam:.2f}", xy=(optimal_lam, total_error[optimal_idx]),
             xytext=(optimal_lam + 0.15, total_error[optimal_idx] + 0.3),
             fontsize=12, color="green", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="green", lw=2))

# 
ax4.axvspan(0.9, 0.97, alpha=0.15, color="gold", label="PPO  (0.9~0.97)")

ax4.set_xlabel("λ ", fontsize=13)
ax4.set_ylabel("", fontsize=13)
ax4.set_title("-（）", fontsize=14, fontweight="bold")
ax4.legend(fontsize=11, loc="center right")
ax4.set_xlim(0, 1)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("output/gae_visualization.png", dpi=150, bbox_inches="tight")
print(": output/gae_visualization.png")
plt.show()


# ==========================================
# ：
# ==========================================
print("\n" + "=" * 60)
print("： (γ, λ) ")
print("=" * 60)

# 
combos = [
    (0.99, 0.0,  ""),
    (0.99, 0.5,  ""),
    (0.99, 0.95, "PPO "),
    (0.99, 1.0,  ""),
    (0.5,  0.95, " + GAE"),
    (1.0,  0.95, " + GAE"),
]

print(f"\n{'':<20} {'γ':>5} {'λ':>5}", end="")
for t in range(n_steps):
    print(f"  {'A(s'+str(t)+')':>8}", end="")
print()
print("-" * (20 + 5 + 5 + 10 * n_steps))

for gamma, lam, desc in combos:
    adv, _ = compute_gae(rewards, values, dones, gamma=gamma, lam=lam)
    print(f"{desc:<20} {gamma:>5.2f} {lam:>5.2f}", end="")
    for t in range(n_steps):
        print(f"  {adv[t]:>8.4f}", end="")
    print()

print("\n" + "=" * 60)
print(":")
print("  1. λ -")
print("  2. γ ")
print("  3. PPO : γ=0.99, λ=0.95")
print("  4. λ=0 →  TD，λ=1 → ")
print("=" * 60)
