"""
5：REINFORCE  —— CartPole-v1
，"，"

：
     —— ，
    ""（）；
    ""（）。

REINFORCE ：
    ∇J(θ) ≈ Σ_t [∇log π(a_t|s_t)] * G_t
     G_t  t 

：
    python reinforce_cartpole.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque

# 
os.makedirs("output", exist_ok=True)

# ，
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# ：（Policy Network）
# ==========================================
class PolicyNetwork(nn.Module):
    """
    ：

    ：4 () → 128 → 128 → 2 ()
     Softmax ，

    CartPole ：[, , , ]
    CartPole ：[, ]
    """

    def __init__(self, state_dim=4, action_dim=2, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),   #  → 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),   #  → 
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),   #  → （logits）
        )

    def forward(self, x):
        """
        ： → 

        ：
            x: ， [batch_size, state_dim]
        ：
            probs: ， [batch_size, action_dim]， Softmax
        """
        logits = self.network(x)
        probs = torch.softmax(logits, dim=-1)
        return probs


# ==========================================
# ：（Returns）
# ==========================================
def compute_returns(rewards, gamma=0.99):
    """
     G_t

    ：G_t = r_t + γ * r_{t+1} + γ² * r_{t+2} + ...

    （gamma=0.99）：
        rewards = [1, 1, 1, 1, 1]
        G_0 = 1 + 0.99*1 + 0.99²*1 + ... ≈ 4.90
        G_4 = 1

    ：
        rewards: 
        gamma: ，1
    ：
        returns: 
    """
    returns = []
    G = 0  # 

    # ： G_t = r_t + gamma * G_{t+1} 
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)  # ，

    return returns


# ==========================================
# ：
# ==========================================
def collect_episode(policy, env):
    """
    ，

    REINFORCE  on-policy ，，
    ，。

    ：
        policy: 
        env: Gymnasium 
    ：
        states: 
        actions: 
        rewards: 
        episode_reward: 
    """
    state, _ = env.reset()
    states, actions, rewards = [], [], []

    done = False
    truncated = False

    while not (done or truncated):
        # 
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  #  batch 

        # 
        with torch.no_grad():
            probs = policy(state_tensor)

        # （！ argmax）
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()

        # ，
        next_state, reward, done, truncated, _ = env.step(action)

        # 
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state

    episode_reward = sum(rewards)
    return states, actions, rewards, episode_reward


# ==========================================
# ：（REINFORCE ）
# ==========================================
def train_one_episode(policy, optimizer, states, actions, returns):
    """
    REINFORCE ：

     = - Σ_t [log π(a_t|s_t) * G_t]

    ：
    ∇loss = - Σ_t [∇log π(a_t|s_t) * G_t] = -∇J(θ)

     minimize loss = maximize J(θ)（）

    ：
        policy: 
        optimizer: 
        states: 
        actions: 
        returns: 
    ：
        loss_value: 
    """
    # 
    states_tensor = torch.FloatTensor(np.array(states))
    actions_tensor = torch.LongTensor(actions)
    returns_tensor = torch.FloatTensor(returns)

    # ：
    probs = policy(states_tensor)

    #  log π(a_t|s_t)
    # gather(1, actions) 
    action_probs = probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
    log_probs = torch.log(action_probs + 1e-8)  #  log(0)

    # ：-log π(a_t|s_t) * G_t
    # ：
    #    G_t > 0（），-log_prob * G_t < 0， log_prob → 
    #    G_t < 0（），-log_prob * G_t > 0， log_prob → 
    loss = -(log_probs * returns_tensor).mean()

    #  + 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# ==========================================
# ：
# ==========================================
def train():
    """
    REINFORCE 

    ：
        - num_episodes = 500： 500 
        - gamma = 0.99：，
        - learning_rate = 1e-3：
        - hidden_dim = 128：
    """
    # ----------  ----------
    num_episodes = 500
    gamma = 0.99
    learning_rate = 1e-3
    hidden_dim = 128

    # ----------  ----------
    env = gym.make("CartPole-v1")
    policy = PolicyNetwork(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dim=hidden_dim,
    )
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    # 
    episode_rewards = []  # 
    episode_losses = []   # 

    print("=" * 60)
    print("  REINFORCE  —— CartPole-v1 ")
    print("=" * 60)
    print(f"  :")
    print(f"    : {num_episodes}")
    print(f"     γ: {gamma}")
    print(f"    : {learning_rate}")
    print(f"    : {hidden_dim}")
    print("=" * 60)

    # ----------  ----------
    for episode in range(num_episodes):
        # ：
        states, actions, rewards, episode_reward = collect_episode(policy, env)

        # ：
        returns = compute_returns(rewards, gamma=gamma)

        # ：
        loss_value = train_one_episode(policy, optimizer, states, actions, returns)

        # 
        episode_rewards.append(episode_reward)
        episode_losses.append(loss_value)

        #  50 
        if (episode + 1) % 50 == 0:
            recent_rewards = episode_rewards[-50:]
            avg_reward = np.mean(recent_rewards)
            print(
                f"   {episode + 1:4d}/{num_episodes} | "
                f": {episode_reward:6.1f} | "
                f" 50 : {avg_reward:6.1f} | "
                f": {loss_value:.4f}"
            )

    env.close()

    # ----------  ----------
    print("=" * 60)
    print("  ！")
    print(f"   50 : {np.mean(episode_rewards[-50:]):.1f}")
    print(f"  : {np.max(episode_rewards):.1f}")
    print("=" * 60)

    # ----------  ----------
    plot_training_curve(episode_rewards)


# ==========================================
# ：
# ==========================================
def plot_training_curve(episode_rewards):
    """
    

    （window=50），
    。
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # （，）
    ax.plot(episode_rewards, alpha=0.3, color='steelblue', label='（）')

    # （，）
    window = 50
    if len(episode_rewards) >= window:
        moving_avg = []
        for i in range(len(episode_rewards)):
            start = max(0, i - window + 1)
            moving_avg.append(np.mean(episode_rewards[start:i + 1]))
        ax.plot(moving_avg, color='crimson', linewidth=2.0,
                label=f'（={window}）')

    ax.set_xlabel('', fontsize=12)
    ax.set_ylabel('', fontsize=12)
    ax.set_title('REINFORCE  —— CartPole-v1 ', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/reinforce_cartpole_rewards.png', dpi=150, bbox_inches='tight')
    print("   output/reinforce_cartpole_rewards.png")
    plt.show()


# ==========================================
# 
# ==========================================
if __name__ == "__main__":
    train()
