"""
3： —— 
：、、ε-、UCB


：
    python two_armed_bandit.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# 
os.makedirs("output", exist_ok=True)


# ==========================================
# ：
# ==========================================
class TwoArmedBandit:
    """
    
    -  A： 0.6（）
    -  B： 0.4

    ，：
    （），（A  B），
    。 MDP ，
    " vs "。
    """

    def __init__(self, prob_a=0.6, prob_b=0.4):
        self.prob_a = prob_a  #  A 
        self.prob_b = prob_b  #  B 
        # ，（regret）
        self.best_prob = max(prob_a, prob_b)

    def pull(self, arm):
        """
        ，（0  1）

        ：
            arm: 0  A，1  B
        ：
            reward: 1 ，0 
        """
        if arm == 0:
            return 1 if np.random.random() < self.prob_a else 0
        else:
            return 1 if np.random.random() < self.prob_b else 0


# ==========================================
# ：
# ==========================================

def strategy_random(bandit, n_steps):
    """
    ：
     A  B，。

    。，
    ：(0.6 + 0.4) / 2 = 0.5
    """
    rewards = []
    for _ in range(n_steps):
        arm = np.random.choice([0, 1])  # 
        reward = bandit.pull(arm)
        rewards.append(reward)
    return np.array(rewards)


def strategy_greedy(bandit, n_steps):
    """
    ：
    。

    ：，，
    ，，。
    ""。
    """
    rewards = []
    # Q[a]  a 
    Q = np.zeros(2)       # 
    counts = np.zeros(2)  # 

    for _ in range(n_steps):
        # （，）
        arm = np.argmax(Q)
        reward = bandit.pull(arm)
        rewards.append(reward)

        # ：
        counts[arm] += 1
        Q[arm] += (reward - Q[arm]) / counts[arm]

    return np.array(rewards)


def strategy_epsilon_greedy(bandit, n_steps, epsilon=0.1):
    """
    ：ε- (ε = 0.1)
     ε ， 1-ε 。

    ε = 0.1  10% 。
    " vs "。
    ε  → ；
    ε  → 。
    """
    rewards = []
    Q = np.zeros(2)
    counts = np.zeros(2)

    for _ in range(n_steps):
        #  ε ，
        if np.random.random() < epsilon:
            arm = np.random.choice([0, 1])  # 
        else:
            arm = np.argmax(Q)  # 

        reward = bandit.pull(arm)
        rewards.append(reward)

        counts[arm] += 1
        Q[arm] += (reward - Q[arm]) / counts[arm]

    return np.array(rewards)


def strategy_ucb(bandit, n_steps, c=2.0):
    """
    ： (UCB, Upper Confidence Bound)
     " + " 。

    UCB ：Q(a) + c * sqrt(ln(t) / N(a))
    - Q(a)： a 
    - c：（ c=2）
    - t：
    - N(a)： a 

    ：（N(a) ），
    ，，
     UCB ""。
    ，，。
    """
    rewards = []
    Q = np.zeros(2)
    counts = np.zeros(2)

    for t in range(1, n_steps + 1):
        # ，
        if t <= 2:
            arm = t - 1
        else:
            #  UCB 
            ucb_values = np.zeros(2)
            for a in range(2):
                # ：，
                uncertainty = c * np.sqrt(np.log(t) / counts[a])
                ucb_values[a] = Q[a] + uncertainty
            arm = np.argmax(ucb_values)

        reward = bandit.pull(arm)
        rewards.append(reward)

        counts[arm] += 1
        Q[arm] += (reward - Q[arm]) / counts[arm]

    return np.array(rewards)


# ==========================================
# ：
# ==========================================
def run_experiment():
    """
    ： n_runs ， n_steps ，
    """
    n_steps = 1000  # 
    n_runs = 200    # （）

    # 
    all_rewards = {
        '': np.zeros(n_steps),
        '': np.zeros(n_steps),
        'ε- (ε=0.1)': np.zeros(n_steps),
        'UCB (c=2)': np.zeros(n_steps),
    }

    print("=" * 60)
    print("  ：")
    print("=" * 60)
    print(f"   A : 0.6（）")
    print(f"   B : 0.4")
    print(f"  : {n_steps}")
    print(f"  : {n_runs}")
    print("-" * 60)
    print("...")

    for run in range(n_runs):
        bandit = TwoArmedBandit(prob_a=0.6, prob_b=0.4)

        # 
        all_rewards[''] += strategy_random(bandit, n_steps)
        all_rewards[''] += strategy_greedy(bandit, n_steps)
        all_rewards['ε- (ε=0.1)'] += strategy_epsilon_greedy(bandit, n_steps)
        all_rewards['UCB (c=2)'] += strategy_ucb(bandit, n_steps)

        if (run + 1) % 50 == 0:
            print(f"   {run + 1}/{n_runs} ...")

    # 
    for key in all_rewards:
        all_rewards[key] /= n_runs

    print("！")
    print()

    # ==========================================
    # ：
    # ==========================================
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1：
    ax1 = axes[0]
    colors = ['#9E9E9E', '#FF9800', '#2196F3', '#4CAF50']
    for (name, rewards), color in zip(all_rewards.items(), colors):
        # 
        cumulative_avg = np.cumsum(rewards) / np.arange(1, n_steps + 1)
        ax1.plot(cumulative_avg, label=name, color=color, alpha=0.85)

    ax1.axhline(y=0.6, color='red', linestyle='--', alpha=0.5, label=' (prob=0.6)')
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label=' (0.5)')
    ax1.set_xlabel('', fontsize=12)
    ax1.set_ylabel('', fontsize=12)
    ax1.set_title('', fontsize=14)
    ax1.legend(fontsize=9, loc='right')
    ax1.set_ylim(0.35, 0.7)
    ax1.grid(True, alpha=0.3)

    # 2：（regret）
    ax2 = axes[1]
    best_prob = 0.6
    for (name, rewards), color in zip(all_rewards.items(), colors):
        #  =  - 
        regret = best_prob - rewards
        cumulative_regret = np.cumsum(regret)
        ax2.plot(cumulative_regret, label=name, color=color, alpha=0.85)

    ax2.set_xlabel('', fontsize=12)
    ax2.set_ylabel('', fontsize=12)
    ax2.set_title('（）', fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/two_armed_bandit_results.png', dpi=150, bbox_inches='tight')
    print(" output/two_armed_bandit_results.png")
    plt.show()

    # ==========================================
    # ：
    # ==========================================
    print()
    print("=" * 60)
    print("  ")
    print("=" * 60)
    print(f"{'':<20s} {'':<15s} {'':<15s} {'':<10s}")
    print("-" * 60)

    for (name, rewards), color in zip(all_rewards.items(), colors):
        cum_avg = np.cumsum(rewards)
        final_avg = cum_avg[-1] / n_steps
        total_reward = np.sum(rewards)
        total_regret = best_prob * n_steps - total_reward
        print(f"{name:<20s} {final_avg:<15.4f} {rewards[-1]:<15.4f} {total_regret:<10.1f}")

    print("-" * 60)
    print()
    print("：")
    print("  - ：， 0.5 （）")
    print("  - ：，")
    print("  - ε-：，")
    print("  - UCB ：，")


if __name__ == "__main__":
    run_experiment()
