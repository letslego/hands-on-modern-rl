"""
5：REINFORCE with Baseline —— 
 REINFORCE （Value Network）

： REINFORCE ，
：
     = G_t - V(s_t)
     V(s_t) 

？
    - G_t （ 200），
    -  V(s) ， 0 ，
    -  E[G_t - b] = E[G_t]（ b ），，

：
    python reinforce_with_baseline.py
"""

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# 
os.makedirs("output", exist_ok=True)
SEED = 0

# 
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# ：
# ==========================================
class PolicyNetwork(nn.Module):
    """
    （Actor）： → 

    ：4 → 128 → 128 → 2（Softmax ）
    """

    def __init__(self, state_dim=4, action_dim=2, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        logits = self.network(x)
        probs = torch.softmax(logits, dim=-1)
        return probs


class ValueNetwork(nn.Module):
    """
    （Baseline/Critic）： → 

    ：4 → 128 → 128 → 1（）
     V(s)， s 

    ""： V(s)， A(s)
    """

    def __init__(self, state_dim=4, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # 
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)  # ， [batch_size]


# ==========================================
# ：
# ==========================================
def compute_returns(rewards, gamma=0.99):
    """
     G_t = r_t + γ * r_{t+1} + γ² * r_{t+2} + ...

    ：
        rewards: 
        gamma: 
    ：
        returns: 
    """
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    return returns


# ==========================================
# ：
# ==========================================
def collect_episode(policy, env):
    """
    

    ：
        policy: 
        env: 
    ：
        states, actions, rewards, episode_reward
    """
    state, _ = env.reset()
    states, actions, rewards = [], [], []
    done, truncated = False, False

    while not (done or truncated):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            probs = policy(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()

        next_state, reward, done, truncated, _ = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state

    episode_reward = sum(rewards)
    return states, actions, rewards, episode_reward


# ==========================================
# ： REINFORCE 
# ==========================================
def train_vanilla_reinforce(num_episodes=500, gamma=0.99, lr=1e-3):
    """
     REINFORCE（）

     = -Σ log π(a_t|s_t) * G_t
     G_t 
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = gym.make("CartPole-v1")
    env.reset(seed=SEED)
    policy = PolicyNetwork(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
    )
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    episode_rewards = []
    gradient_estimates = []  # ，

    for episode in range(num_episodes):
        # 
        states, actions, rewards, episode_reward = collect_episode(policy, env)

        # 
        returns = compute_returns(rewards, gamma)

        # 
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions)
        returns_t = torch.FloatTensor(returns)

        # 
        probs = policy(states_t)
        action_probs = probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        log_probs = torch.log(action_probs + 1e-8)

        # 
        loss = -(log_probs * returns_t).mean()

        # （）
        with torch.no_grad():
            grad_estimate = (log_probs * returns_t).mean().item()
            gradient_estimates.append(grad_estimate)

        # 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_rewards.append(episode_reward)

        if (episode + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-50:])
            print(f"  [Vanilla]  {episode+1:4d} | 50: {avg:6.1f}")

    env.close()
    return episode_rewards, gradient_estimates


# ==========================================
# ：REINFORCE with Baseline 
# ==========================================
def train_reinforce_with_baseline(num_episodes=500, gamma=0.99, lr=1e-3):
    """
    REINFORCE + 

    ：A(s,a) = G_t - V(s_t)
    ：-Σ log π(a_t|s_t) * A(s_t, a_t)
    ：MSE(V(s_t), G_t)

    ：
        - ""（）
        - ""（）
    """
    baseline_seed = SEED + 100
    random.seed(baseline_seed)
    np.random.seed(baseline_seed)
    torch.manual_seed(baseline_seed)

    env = gym.make("CartPole-v1")
    env.reset(seed=baseline_seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 
    policy = PolicyNetwork(state_dim=state_dim, action_dim=action_dim)
    value_net = ValueNetwork(state_dim=state_dim)

    # 
    policy_optimizer = optim.Adam(policy.parameters(), lr=lr)
    value_optimizer = optim.Adam(value_net.parameters(), lr=lr)

    episode_rewards = []
    gradient_estimates = []  # 

    for episode in range(num_episodes):
        # 
        states, actions, rewards, episode_reward = collect_episode(policy, env)

        # 
        returns = compute_returns(rewards, gamma)

        # 
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions)
        returns_t = torch.FloatTensor(returns)

        # ========== （Critic） ==========
        # ： V(s) ≈ G_t
        values = value_net(states_t)
        value_loss = nn.MSELoss()(values, returns_t)

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        # ==========  ==========
        #  =  - 
        # A > 0 "" → 
        # A < 0 "" → 
        with torch.no_grad():
            values_pred = value_net(states_t)
        advantages = returns_t - values_pred

        # ========== （Actor） ==========
        probs = policy(states_t)
        action_probs = probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        log_probs = torch.log(action_probs + 1e-8)

        # ：
        policy_loss = -(log_probs * advantages).mean()

        # （）
        with torch.no_grad():
            grad_estimate = (log_probs * advantages).mean().item()
            gradient_estimates.append(grad_estimate)

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        episode_rewards.append(episode_reward)

        if (episode + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-50:])
            print(f"  [Value Baseline]  {episode+1:4d} | 50: {avg:6.1f}")

    env.close()
    return episode_rewards, gradient_estimates


# ==========================================
# ：
# ==========================================
def run_comparison():
    """
    ：Vanilla REINFORCE vs REINFORCE + Value Baseline

    ：
        1. （）
        2. （，）
    """
    num_episodes = 500
    gamma = 0.99
    lr = 1e-3

    print("=" * 60)
    print("  REINFORCE ")
    print("=" * 60)
    print(f"  : {num_episodes}")
    print(f"   γ: {gamma}")
    print(f"  : {lr}")
    print("=" * 60)

    # ---------- 1： REINFORCE ----------
    print("\n[1]  Vanilla REINFORCE（）...")
    vanilla_rewards, vanilla_grads = train_vanilla_reinforce(
        num_episodes=num_episodes, gamma=gamma, lr=lr
    )

    # ---------- 2：REINFORCE + Value Baseline ----------
    print("\n[2]  REINFORCE + Value Baseline（）...")
    baseline_rewards, baseline_grads = train_reinforce_with_baseline(
        num_episodes=num_episodes, gamma=gamma, lr=lr
    )

    # ----------  ----------
    print("\n" + "=" * 60)
    print("  ")
    print("=" * 60)

    vanilla_grad_var = np.var(vanilla_grads)
    baseline_grad_var = np.var(baseline_grads)

    print(f"  Vanilla REINFORCE : {vanilla_grad_var:.6f}")
    print(f"  REINFORCE+Value Baseline : {baseline_grad_var:.6f}")

    if vanilla_grad_var > 0:
        ratio = vanilla_grad_var / max(baseline_grad_var, 1e-10)
        print(f"  （Vanilla/Value Baseline）: {ratio:.2f}x")
        print(f"  Value Baseline  {1/ratio*100:.1f}%")

    print(f"\n  Vanilla REINFORCE 50: {np.mean(vanilla_rewards[-50:]):.1f}")
    print(f"  REINFORCE+Value Baseline 50: {np.mean(baseline_rewards[-50:]):.1f}")
    print("=" * 60)

    # ---------- 1： ----------
    plot_reward_comparison(vanilla_rewards, baseline_rewards, window=50)

    # ---------- 2： ----------
    plot_variance_comparison(vanilla_grads, baseline_grads, window=50)


# ==========================================
# ：
# ==========================================
def plot_reward_comparison(vanilla_rewards, baseline_rewards, window=50):
    """
    

    ，：
        - Value Baseline 
        - Value Baseline （）
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Vanilla REINFORCE
    ax.plot(vanilla_rewards, alpha=0.2, color='steelblue')
    vanilla_avg = [np.mean(vanilla_rewards[max(0, i-window+1):i+1])
                   for i in range(len(vanilla_rewards))]
    ax.plot(vanilla_avg, color='steelblue', linewidth=2.0,
            label='Vanilla REINFORCE')

    # REINFORCE + Value Baseline
    ax.plot(baseline_rewards, alpha=0.2, color='crimson')
    baseline_avg = [np.mean(baseline_rewards[max(0, i-window+1):i+1])
                    for i in range(len(baseline_rewards))]
    ax.plot(baseline_avg, color='crimson', linewidth=2.0,
            label='REINFORCE + Value Baseline')

    ax.set_xlabel('', fontsize=12)
    ax.set_ylabel('', fontsize=12)
    ax.set_title('REINFORCE （Vanilla vs Value Baseline）', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/reinforce_baseline_reward_comparison.png', dpi=150, bbox_inches='tight')
    print("   output/reinforce_baseline_reward_comparison.png")
    plt.show()


# ==========================================
# ：
# ==========================================
def plot_variance_comparison(vanilla_grads, baseline_grads, window=50):
    """
    

    ： Value Baseline 。
    ，，。
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # 
    def moving_variance(data, w):
        variances = []
        for i in range(len(data)):
            start = max(0, i - w + 1)
            variances.append(np.var(data[start:i + 1]))
        return variances

    vanilla_var = moving_variance(vanilla_grads, window)
    baseline_var = moving_variance(baseline_grads, window)

    ax.plot(vanilla_var, color='steelblue', linewidth=1.5, alpha=0.8,
            label='Vanilla REINFORCE')
    ax.plot(baseline_var, color='crimson', linewidth=1.5, alpha=0.8,
            label='REINFORCE + Value Baseline')

    ax.set_xlabel('', fontsize=12)
    ax.set_ylabel(f'（={window}）', fontsize=12)
    ax.set_title(' —— Value Baseline ', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # ，
    if len(vanilla_var) > 100:
        mid_point = len(vanilla_var) // 2
        ax.annotate(
            'Value Baseline ',
            xy=(mid_point, baseline_var[mid_point]),
            xytext=(mid_point + 50, max(vanilla_var) * 0.7),
            fontsize=11,
            arrowprops=dict(arrowstyle='->', color='gray'),
            color='gray',
        )

    plt.tight_layout()
    plt.savefig('output/reinforce_baseline_variance_comparison.png', dpi=150, bbox_inches='tight')
    print("   output/reinforce_baseline_variance_comparison.png")
    plt.show()


# ==========================================
# 
# ==========================================
if __name__ == "__main__":
    run_comparison()
