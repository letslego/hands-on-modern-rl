"""
4：Q (Deep Q-Network, DQN) —— 
 CartPole-v1  DQN 

：
     Q(s, a)，。
     2015  DeepMind ，。

：
    1. Q ： (MLP) 
    2. ：，
    3. ：，
    4. ε-：

：
    python dqn_cartpole.py
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
# ：Q 
# ==========================================
class QNetwork(nn.Module):
    """
    Q ： Q 
    ：state_dim → 128 → 128 → action_dim

    （CartPole  4 ），
     Q （CartPole  2 ，/）。
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),   # ： → 128
            nn.ReLU(),
            nn.Linear(128, 128),          # ：128 → 128
            nn.ReLU(),
            nn.Linear(128, action_dim),   # ：128 → 
        )

    def forward(self, x):
        """：， Q """
        return self.net(x)


# ==========================================
# ：
# ==========================================
class ReplayBuffer:
    """
    ：

    ？
    -  (i.i.d.)
    - ， (s, a, r, s') 
    - 
    - （）
    """

    def __init__(self, capacity=10000):
        """ deque ，"""
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """ (s, a, r, s', done)"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """，"""
        batch = random.sample(self.buffer, batch_size)
        #  numpy ， tensor
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
# ：DQN 
# ==========================================
class DQNAgent:
    """
    DQN ： Q 、、 ε-

    DQN ：
    1.  (Experience Replay)：
    2.  (Target Network)：
    3. ε- (Epsilon-Greedy)：
    """

    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.action_dim = action_dim
        self.gamma = gamma  # ：

        # Q ：
        self.q_net = QNetwork(state_dim, action_dim)
        # ： Q ，
        self.target_net = QNetwork(state_dim, action_dim)
        #  Q 
        self.target_net.load_state_dict(self.q_net.state_dict())
        # 
        self.target_net.eval()

        # ： Q 
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # 
        self.buffer = ReplayBuffer(capacity=10000)

    def select_action(self, state, epsilon):
        """
        ε-

         ε （），
         1-ε  Q （）。
        """
        if random.random() < epsilon:
            # ：
            return random.randint(0, self.action_dim - 1)
        else:
            # ： Q 
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  #  batch 
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def update(self, batch_size):
        """
         Q 

         DQN ：
            target = r + γ * max_a' Q_target(s', a')
            loss = (target - Q(s, a))²

        ： target_net ，Q  q_net ，
        ""。
        """
        if len(self.buffer) < batch_size:
            return 0.0  # 

        # 
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        #  PyTorch 
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        #  Q(s, a)：
        # gather  actions  Q 
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # ：r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            #  Q 
            next_q_max = self.target_net(next_states).max(dim=1)[0]
            # done = 1 ，
            targets = rewards + self.gamma * next_q_max * (1 - dones)

        # 
        loss = nn.MSELoss()(q_values, targets)

        # 
        self.optimizer.zero_grad()
        loss.backward()
        # ：
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10)
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        """ Q （）"""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        """"""
        torch.save(self.q_net.state_dict(), path)
        print(f" {path}")


# ==========================================
# ：
# ==========================================
def train():
    """ DQN """

    # 
    NUM_EPISODES = 500       # 
    BATCH_SIZE = 64          # 
    EPSILON_START = 1.0      # 
    EPSILON_END = 0.01       # 
    EPSILON_DECAY = 0.995    # 
    TARGET_UPDATE_FREQ = 10  # （ N ）

    # 
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]   # 4
    action_dim = env.action_space.n               # 2
    agent = DQNAgent(state_dim, action_dim, lr=1e-3, gamma=0.99)

    print("=" * 60)
    print("  Q (DQN) —— CartPole-v1 ")
    print("=" * 60)
    print(f"  : {state_dim}")
    print(f"  : {action_dim}")
    print(f"  : {NUM_EPISODES}")
    print(f"  : {BATCH_SIZE}")
    print(f"  : {EPSILON_START}")
    print(f"  :  {TARGET_UPDATE_FREQ} ")
    print("=" * 60)

    # 
    reward_history = []
    epsilon = EPSILON_START

    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        episode_reward = 0

        while True:
            # （ε-）
            action = agent.select_action(state, epsilon)
            # 
            next_state, reward, done, truncated, _ = env.step(action)
            # 
            agent.buffer.push(state, action, reward, next_state, float(done))
            #  Q 
            agent.update(BATCH_SIZE)

            state = next_state
            episode_reward += reward

            if done or truncated:
                break

        # 
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        reward_history.append(episode_reward)

        # 
        if (episode + 1) % TARGET_UPDATE_FREQ == 0:
            agent.update_target()

        #  50 
        if (episode + 1) % 50 == 0:
            recent = reward_history[-50:]
            avg_reward = np.mean(recent)
            print(
                f"   {episode + 1:4d}/{NUM_EPISODES} | "
                f"(50): {avg_reward:6.1f} | "
                f"ε: {epsilon:.3f}"
            )

    env.close()

    # ==========================================
    # ：
    # ==========================================
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(reward_history, alpha=0.3, color='steelblue', label='')

    # ，
    window = 20
    if len(reward_history) >= window:
        moving_avg = [
            np.mean(reward_history[max(0, i - window): i + 1])
            for i in range(len(reward_history))
        ]
        ax.plot(moving_avg, color='red', linewidth=2,
                label=f'{window} ')

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('DQN  —— CartPole-v1')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/dqn_cartpole_training.png", dpi=150)
    print("\n output/dqn_cartpole_training.png")
    plt.show()

    # ==========================================
    # ：
    # ==========================================
    print("\n" + "=" * 60)
    print("  ： 10 ")
    print("=" * 60)

    test_env = gym.make("CartPole-v1")
    test_rewards = []

    for ep in range(10):
        state, _ = test_env.reset()
        total_reward = 0

        while True:
            # ，
            action = agent.select_action(state, epsilon=0.0)
            state, reward, done, truncated, _ = test_env.step(action)
            total_reward += reward

            if done or truncated:
                break

        test_rewards.append(total_reward)
        print(f"   {ep + 1:2d}:  = {total_reward:.0f}")

    print("-" * 60)
    print(f"  : {np.mean(test_rewards):.1f} ± {np.std(test_rewards):.1f}")
    print("=" * 60)

    test_env.close()

    # 
    agent.save("output/dqn_cartpole.pth")

    return reward_history


# ==========================================
# 
# ==========================================
if __name__ == "__main__":
    train()
