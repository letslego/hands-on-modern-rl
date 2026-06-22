"""
4：Double DQN ——  DQN 
 CartPole-v1  DQN  Double DQN

：
     DQN ， Q ：
        target = r + γ * max_a Q_target(s', a)
     Q  (overestimation)。

 —— Double DQN (Hasselt et al., 2016)：
    """ Q "：
    1.  Q ：a* = argmax_a Q(s', a)
    2.  Q ：Q_target(s', a*)
    ：target = r + γ * Q_target(s', argmax_a Q(s', a))

    ，。

：
    python double_dqn_cartpole.py
"""

import os
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt

# 
os.makedirs("output", exist_ok=True)


# ==========================================
# ：Q （ DQN ）
# ==========================================
class QNetwork(nn.Module):
    """
    Q ： Q 
    ：state_dim → 128 → 128 → action_dim
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# ：（ DQN ）
# ==========================================
class ReplayBuffer:
    """："""

    def __init__(self, capacity=10000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ==========================================
# ： DQN 
# ==========================================
class DQNAgent:
    """
     DQN 

    ：
        target = r + γ * max_a' Q_target(s', a')

    ：max """ Q "，
    。
    """

    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.action_dim = action_dim
        self.gamma = gamma

        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(capacity=10000)

    def select_action(self, state, epsilon):
        """ε-"""
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def update(self, batch_size):
        """
         DQN 

        ：r + γ * Q_target(s').max()
         Q 。
        """
        if len(self.buffer) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        #  Q 
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        #  DQN ：
        with torch.no_grad():
            next_q_max = self.target_net(next_states).max(dim=1)[0]
            targets = rewards + self.gamma * next_q_max * (1 - dones)

        loss = nn.MSELoss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10)
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        """"""
        self.target_net.load_state_dict(self.q_net.state_dict())


# ==========================================
# ：Double DQN 
# ==========================================
class DoubleDQNAgent(DQNAgent):
    """
    Double DQN （ DQN）

    ：
         DQN：target = r + γ * Q_target(s').max()
        Double DQN：target = r + γ * Q_target(s')[argmax_a Q(s')]

    ：
    - Q ""（）
    - ""（ Q ）
    - ，
    """

    def update(self, batch_size):
        """
        Double DQN （！）

        ：
        1.  q_net ：a* = argmax_a q_net(s')
        2.  target_net  Q ：Q_target(s', a*)
        3. ：target = r + γ * Q_target(s', a*)
        """
        if len(self.buffer) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        #  Q （ DQN ）
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # ★ Double DQN ： ★
        with torch.no_grad():
            # ： Q 
            # q_net  Q ， argmax 
            best_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)

            # ： Q 
            # target_net  Q 
            next_q_values = self.target_net(next_states).gather(1, best_actions).squeeze(1)

            # ：
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10)
        self.optimizer.step()

        return loss.item()


# ==========================================
# ：
# ==========================================
def train_agent(agent, num_episodes=300, batch_size=64,
                epsilon_start=1.0, epsilon_end=0.01,
                epsilon_decay=0.995, target_update_freq=10):
    """
    ， DQN  Double DQN

    ：
        agent: DQN  Double DQN 
        

    ：
        reward_history: 
    """
    env = gym.make("CartPole-v1")
    reward_history = []
    epsilon = epsilon_start

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        while True:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.buffer.push(state, action, reward, next_state, float(done))
            agent.update(batch_size)

            state = next_state
            episode_reward += reward

            if done or truncated:
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        reward_history.append(episode_reward)

        if (episode + 1) % target_update_freq == 0:
            agent.update_target()

    env.close()
    return reward_history


# ==========================================
# ：
# ==========================================
def main():
    """ DQN  Double DQN """

    # 
    NUM_EPISODES = 300
    BATCH_SIZE = 64
    LR = 1e-3
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    TARGET_UPDATE_FREQ = 10

    # 
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]   # 4
    action_dim = env.action_space.n               # 2
    env.close()

    print("=" * 60)
    print("  DQN vs Double DQN  —— CartPole-v1")
    print("=" * 60)
    print(f"  : {state_dim}")
    print(f"  : {action_dim}")
    print(f"  : {NUM_EPISODES}")
    print(f"  : {BATCH_SIZE}")
    print(f"  : {LR}")
    print(f"  : {GAMMA}")
    print("=" * 60)

    # ------------------------------------------
    #  DQN
    # ------------------------------------------
    print("\n[1/2]  DQN...")
    print("-" * 60)

    dqn_agent = DQNAgent(state_dim, action_dim, lr=LR, gamma=GAMMA)
    dqn_rewards = train_agent(
        dqn_agent,
        num_episodes=NUM_EPISODES,
        batch_size=BATCH_SIZE,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        target_update_freq=TARGET_UPDATE_FREQ,
    )

    dqn_avg = np.mean(dqn_rewards[-50:])
    print(f"  DQN ！ 50 : {dqn_avg:.1f}")

    # ------------------------------------------
    #  Double DQN
    # ------------------------------------------
    print("\n[2/2]  Double DQN...")
    print("-" * 60)

    double_dqn_agent = DoubleDQNAgent(state_dim, action_dim, lr=LR, gamma=GAMMA)
    double_dqn_rewards = train_agent(
        double_dqn_agent,
        num_episodes=NUM_EPISODES,
        batch_size=BATCH_SIZE,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        target_update_freq=TARGET_UPDATE_FREQ,
    )

    ddqn_avg = np.mean(double_dqn_rewards[-50:])
    print(f"  Double DQN ！ 50 : {ddqn_avg:.1f}")

    # ==========================================
    # ：
    # ==========================================
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- 1： ---
    ax1.plot(dqn_rewards, alpha=0.3, color='steelblue')
    ax1.plot(double_dqn_rewards, alpha=0.3, color='coral')

    # 
    window = 20
    dqn_ma = [np.mean(dqn_rewards[max(0, i - window): i + 1])
              for i in range(len(dqn_rewards))]
    ddqn_ma = [np.mean(double_dqn_rewards[max(0, i - window): i + 1])
               for i in range(len(double_dqn_rewards))]

    ax1.plot(dqn_ma, color='steelblue', linewidth=2, label='DQN')
    ax1.plot(ddqn_ma, color='coral', linewidth=2, label='Double DQN')
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.set_title('DQN vs Double DQN ')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- 2：（） ---
    ax2.plot(dqn_ma, color='steelblue', linewidth=2, label='DQN')
    ax2.plot(ddqn_ma, color='coral', linewidth=2, label='Double DQN')
    ax2.fill_between(
        range(len(dqn_ma)), dqn_ma, ddqn_ma,
        where=[d > dd for d, dd in zip(dqn_ma, ddqn_ma)],
        alpha=0.15, color='steelblue', label='DQN '
    )
    ax2.fill_between(
        range(len(dqn_ma)), dqn_ma, ddqn_ma,
        where=[dd >= d for d, dd in zip(dqn_ma, ddqn_ma)],
        alpha=0.15, color='coral', label='Double DQN '
    )
    ax2.set_xlabel('')
    ax2.set_ylabel('（）')
    ax2.set_title(f'{window} ')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/dqn_vs_double_dqn.png", dpi=150)
    print("\n output/dqn_vs_double_dqn.png")
    plt.show()

    # ==========================================
    # ：
    # ==========================================
    print("\n" + "=" * 60)
    print("  ")
    print("=" * 60)

    print(f"\n   DQN:")
    print(f"     50 : {dqn_avg:.1f}")
    print(f"    : {max(dqn_rewards):.0f}")

    print(f"\n  Double DQN:")
    print(f"     50 : {ddqn_avg:.1f}")
    print(f"    : {max(double_dqn_rewards):.0f}")

    print(f"\n   (Double DQN - DQN): {ddqn_avg - dqn_avg:+.1f}")

    print("\n" + "-" * 60)
    print("  ：")
    print("   DQN：target = r + γ * Q_target(s').max()")
    print("  Double DQN：target = r + γ * Q_target(s')[Q(s').argmax()]")
    print("  ↑ ，")
    print("=" * 60)


# ==========================================
# 
# ==========================================
if __name__ == "__main__":
    main()
