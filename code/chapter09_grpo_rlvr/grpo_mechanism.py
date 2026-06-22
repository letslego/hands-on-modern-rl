"""
9：GRPO  —— 
==========================================================

 GRPO (Group Relative Policy Optimization) ：
  1. （group），
  2. 
  3. （advantage）
  4.  PPO  Critic 
  5. 

GRPO ：
  -  Value Network（Critic）
  - 
  - ，

：
  advantage_i = (reward_i - mean(rewards)) / (std(rewards) + eps)

：
  python grpo_mechanism.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# 
os.makedirs("output", exist_ok=True)

# ，
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# ：GRPO 
# ==========================================
def compute_grpo_advantages(rewards):
    """
    GRPO ：，

    ：
        mean_r = mean(rewards)
        std_r  = std(rewards) + eps   (eps )
        advantage_i = (reward_i - mean_r) / std_r

    ：
        - advantage > 0 ""，
        - advantage < 0 ""，
        -  0 ， 1

    ：
        rewards: numpy ，
    ：
        advantages: numpy ，
    """
    eps = 1e-8  # ， std=0 
    mean_r = rewards.mean()
    std_r = rewards.std() + eps
    advantages = (rewards - mean_r) / std_r
    return advantages


def compute_ppo_advantages(rewards, value_predictions):
    """
    PPO ： Critic 

    ：
        advantage_i = reward_i - V(s_i)

     Value Network  V(s)，
    ，。

    ：
        rewards: numpy ，
        value_predictions: numpy ，Critic 
    ：
        advantages: numpy ，Critic 
    """
    advantages = rewards - value_predictions
    return advantages


# ==========================================
# ：
# ==========================================
def generate_synthetic_rewards(group_size=8, seed=42):
    """
    ， GRPO 

    ：
        ， group_size ，
        （）。

     [0, 1] ，：
        - 0.0 ~ 0.3：
        - 0.3 ~ 0.7：
        - 0.7 ~ 1.0：

    ：
        group_size: （GRPO  8~16）
        seed: ，
    ：
        rewards: numpy ， (group_size,)
    """
    np.random.seed(seed)
    # ：，
    rewards = np.array([0.35, 0.52, 0.68, 0.41, 0.89, 0.73, 0.28, 0.61])
    return rewards


def simulate_critic_predictions(rewards, noise_scale=0.08, seed=123):
    """
     PPO  Critic 

     Critic ，：
        V(s) ≈ reward + noise
    Critic 。

    ：
        rewards: 
        noise_scale: ， Critic 
        seed: 
    ：
        value_predictions: numpy ， Critic 
    """
    np.random.seed(seed)
    noise = np.random.normal(0, noise_scale, size=rewards.shape)
    value_predictions = rewards + noise
    # 
    value_predictions = np.clip(value_predictions, 0.0, 1.0)
    return value_predictions


# ==========================================
# ： GRPO 
# ==========================================
def demonstrate_grpo_step_by_step():
    """
     GRPO ，

    ：
        Step 1: 
        Step 2: （）
        Step 3:  GRPO 
        Step 4: 
        Step 5:  PPO  Critic 
    """
    group_size = 8

    print("=" * 70)
    print("  GRPO  —— ")
    print("=" * 70)

    # ---------- Step 1： ----------
    rewards = generate_synthetic_rewards(group_size=group_size)
    print(f"\n【Step 1】（group_size = {group_size}）")
    print("-" * 70)
    print("  ：， 8 ，")
    print()
    for i, r in enumerate(rewards):
        bar = "|" * int(r * 30)  # 
        print(f"   {i+1}: reward = {r:.2f}  {bar}")

    # ---------- Step 2： ----------
    mean_r = rewards.mean()
    std_r = rewards.std()
    print(f"\n【Step 2】")
    print("-" * 70)
    print(f"   mean(r) = {mean_r:.4f}")
    print(f"   std(r) = {std_r:.4f}")
    print(f"  std(r) + 1e-8 = {std_r + 1e-8:.4f}  （ epsilon ）")

    # ---------- Step 3：GRPO  ----------
    grpo_advantages = compute_grpo_advantages(rewards)
    print(f"\n【Step 3】GRPO  = (reward - mean) / (std + eps)")
    print("-" * 70)
    for i in range(group_size):
        adv = grpo_advantages[i]
        sign = "+" if adv >= 0 else ""
        print(f"   {i+1}: ({rewards[i]:.2f} - {mean_r:.4f}) / {std_r + 1e-8:.4f}"
              f" = {sign}{adv:.4f}")

    print()
    print(f"  : {grpo_advantages.mean():.6f}  （ 0）")
    print(f"  : {grpo_advantages.std():.6f}  （ 1）")

    # ---------- Step 4： ----------
    sorted_indices = np.argsort(grpo_advantages)[::-1]  # 
    print(f"\n【Step 4】 GRPO （）")
    print("-" * 70)
    print(f"  {'':>4s}  {'':>4s}  {'':>8s}  {'GRPO':>10s}  {''}")
    print(f"  {'----':>4s}  {'----':>4s}  {'--------':>8s}  {'----------':>10s}  {'----'}")
    for rank, idx in enumerate(sorted_indices):
        adv = grpo_advantages[idx]
        if adv > 0.5:
            interpretation = "，"
        elif adv > 0:
            interpretation = "，"
        elif adv > -0.5:
            interpretation = "，"
        else:
            interpretation = "，"
        print(f"  {rank+1:>4d}  {idx+1:>4d}  {rewards[idx]:>8.4f}  {adv:>+10.4f}  {interpretation}")

    # ---------- Step 5： PPO  ----------
    value_preds = simulate_critic_predictions(rewards)
    ppo_advantages = compute_ppo_advantages(rewards, value_preds)
    print(f"\n【Step 5】GRPO vs PPO ")
    print("-" * 70)
    print(f"  {'':>4s}  {'':>8s}  {'Critic':>10s}  "
          f"{'PPO':>10s}  {'GRPO':>10s}  {'':>8s}")
    print(f"  {'----':>4s}  {'--------':>8s}  {'----------':>10s}  "
          f"{'----------':>10s}  {'----------':>10s}  {'--------':>8s}")
    for i in range(group_size):
        diff = grpo_advantages[i] - ppo_advantages[i]
        print(f"  {i+1:>4d}  {rewards[i]:>8.4f}  {value_preds[i]:>10.4f}  "
              f"{ppo_advantages[i]:>+10.4f}  {grpo_advantages[i]:>+10.4f}  {diff:>+8.4f}")

    print()
    print(f"  PPO   → : {ppo_advantages.mean():.4f}, : {ppo_advantages.std():.4f}")
    print(f"  GRPO  → : {grpo_advantages.mean():.6f}, : {grpo_advantages.std():.4f}")
    print()
    print("  ：")
    print("    PPO   = reward - V(s)， Critic ")
    print("    GRPO  = (reward - mean) / std，， Critic")

    return rewards, grpo_advantages, ppo_advantages


# ==========================================
# ： —— 
# ==========================================
def plot_reward_normalization(rewards, grpo_advantages):
    """
    

    ：（0~1 ）
    ：GRPO （ 0 ）
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    group_size = len(rewards)
    x = np.arange(group_size)

    # ：
    colors_raw = []
    for r in rewards:
        if r >= 0.7:
            colors_raw.append('#2ecc71')   # ：
        elif r >= 0.4:
            colors_raw.append('#f39c12')   # ：
        else:
            colors_raw.append('#e74c3c')   # ：

    axes[0].bar(x, rewards, color=colors_raw, edgecolor='white', linewidth=1.5)
    axes[0].axhline(y=rewards.mean(), color='black', linestyle='--',
                     linewidth=1.5, label=f' = {rewards.mean():.3f}')
    axes[0].set_xlabel('', fontsize=12)
    axes[0].set_ylabel('', fontsize=12)
    axes[0].set_title('：', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'#{i+1}' for i in range(group_size)])
    axes[0].legend(fontsize=10)
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, alpha=0.3, axis='y')

    # ：GRPO 
    colors_adv = []
    for a in grpo_advantages:
        if a > 0:
            colors_adv.append('#27ae60')   # ：（）
        else:
            colors_adv.append('#c0392b')   # ：（）

    axes[1].bar(x, grpo_advantages, color=colors_adv, edgecolor='white', linewidth=1.5)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1.0)
    axes[1].set_xlabel('', fontsize=12)
    axes[1].set_ylabel('GRPO ', fontsize=12)
    axes[1].set_title('：GRPO ', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'#{i+1}' for i in range(group_size)])
    axes[1].grid(True, alpha=0.3, axis='y')

    # 
    axes[1].annotate(' → ', xy=(4.5, grpo_advantages[4]),
                      xytext=(5.5, grpo_advantages[4] + 0.3),
                      fontsize=10, color='#27ae60',
                      arrowprops=dict(arrowstyle='->', color='#27ae60'))
    axes[1].annotate(' → ', xy=(6, grpo_advantages[6]),
                      xytext=(0.5, grpo_advantages[6] - 0.4),
                      fontsize=10, color='#c0392b',
                      arrowprops=dict(arrowstyle='->', color='#c0392b'))

    plt.suptitle('GRPO ', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('output/grpo_reward_normalization.png', dpi=150, bbox_inches='tight')
    print("   output/grpo_reward_normalization.png")
    plt.show()


# ==========================================
# ： —— GRPO vs PPO 
# ==========================================
def plot_advantage_comparison(grpo_advantages, ppo_advantages):
    """
     GRPO  PPO 

    ：
    ：
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    group_size = len(grpo_advantages)
    x = np.arange(group_size)
    bar_width = 0.35

    # ：
    bars1 = axes[0].bar(x - bar_width/2, grpo_advantages, bar_width,
                        label='GRPO （）', color='#3498db',
                        edgecolor='white', linewidth=1.0)
    bars2 = axes[0].bar(x + bar_width/2, ppo_advantages, bar_width,
                        label='PPO （Critic ）', color='#e67e22',
                        edgecolor='white', linewidth=1.0)
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[0].set_xlabel('', fontsize=12)
    axes[0].set_ylabel('', fontsize=12)
    axes[0].set_title('GRPO vs PPO （）', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'#{i+1}' for i in range(group_size)])
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')

    # ：
    bins = np.linspace(-2, 2, 20)
    axes[1].hist(grpo_advantages, bins=bins, alpha=0.6, color='#3498db',
                 label='GRPO ', edgecolor='white')
    axes[1].hist(ppo_advantages, bins=bins, alpha=0.6, color='#e67e22',
                 label='PPO ', edgecolor='white')
    axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1.0)
    axes[1].set_xlabel('', fontsize=12)
    axes[1].set_ylabel('', fontsize=12)
    axes[1].set_title('GRPO vs PPO ', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # 
    stats_text = (
        f"GRPO: ={grpo_advantages.mean():.4f}, ={grpo_advantages.std():.4f}\n"
        f"PPO:  ={ppo_advantages.mean():.4f}, ={ppo_advantages.std():.4f}"
    )
    axes[1].text(0.02, 0.95, stats_text, transform=axes[1].transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('output/grpo_vs_ppo_advantages.png', dpi=150, bbox_inches='tight')
    print("   output/grpo_vs_ppo_advantages.png")
    plt.show()


# ==========================================
# ： ——  GRPO 
# ==========================================
def run_multiple_groups(num_groups=5, group_size=8):
    """
    ， GRPO 

     GRPO  PPO ：
        -  0
        -  1
    """
    print("\n" + "=" * 70)
    print("  ：GRPO ")
    print("=" * 70)
    print()
    print(f"  ：{num_groups} ， {group_size} ")
    print()

    grpo_stats = {"mean": [], "std": []}
    ppo_stats = {"mean": [], "std": []}

    for g in range(num_groups):
        # 
        np.random.seed(g * 10 + 7)
        # ：
        base = np.random.uniform(0.2, 0.8)
        spread = np.random.uniform(0.1, 0.3)
        rewards = np.clip(np.random.normal(base, spread, size=group_size), 0.0, 1.0)

        #  GRPO 
        grpo_adv = compute_grpo_advantages(rewards)
        grpo_stats["mean"].append(grpo_adv.mean())
        grpo_stats["std"].append(grpo_adv.std())

        #  PPO 
        value_preds = simulate_critic_predictions(rewards, noise_scale=0.1, seed=g * 5 + 3)
        ppo_adv = compute_ppo_advantages(rewards, value_preds)
        ppo_stats["mean"].append(ppo_adv.mean())
        ppo_stats["std"].append(ppo_adv.std())

        print(f"   {g+1} ：")
        print(f"    : mean={rewards.mean():.4f}, std={rewards.std():.4f}")
        print(f"    GRPO : mean={grpo_adv.mean():.6f}, std={grpo_adv.std():.4f}")
        print(f"    PPO  : mean={ppo_adv.mean():.4f}, std={ppo_adv.std():.4f}")

    print()
    print("  【】")
    print(f"  GRPO  → : {np.mean(grpo_stats['mean']):.6f}, "
          f": {np.std(grpo_stats['mean']):.6f}")
    print(f"  GRPO  → : {np.mean(grpo_stats['std']):.4f}, "
          f": {np.std(grpo_stats['std']):.4f}")
    print()
    print(f"  PPO   → : {np.mean(ppo_stats['mean']):.4f}, "
          f": {np.std(ppo_stats['mean']):.4f}")
    print(f"  PPO   → : {np.mean(ppo_stats['std']):.4f}, "
          f": {np.std(ppo_stats['std']):.4f}")
    print()
    print("  ：GRPO  0（）， 1。")
    print("   PPO  Critic ，。")


# ==========================================
# 
# ==========================================
if __name__ == "__main__":
    #  GRPO 
    rewards, grpo_advantages, ppo_advantages = demonstrate_grpo_step_by_step()

    # ：
    print("\n" + "=" * 70)
    print("  ...")
    print("=" * 70)
    plot_reward_normalization(rewards, grpo_advantages)

    # ：GRPO vs PPO 
    plot_advantage_comparison(grpo_advantages, ppo_advantages)

    # 
    run_multiple_groups()

    # 
    print("\n" + "=" * 70)
    print("  GRPO ")
    print("=" * 70)
    print("""
  GRPO (Group Relative Policy Optimization) ：

  1. ，（group）
  2. （）
  3. ：
     advantage = (reward - mean) / (std + eps)
  4. 

   PPO ：
    -  Critic（Value Network），
    -  0、 1
    - ，

   DPO ：
    - （chosen/rejected）
    - （/）
    - （RLVR）

  DeepSeek-R1  GRPO + RLVR 。
    """)
