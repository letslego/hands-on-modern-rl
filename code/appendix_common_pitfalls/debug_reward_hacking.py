"""
A - ：（Reward Hacking）

、：。

？
    ""——，
    。

：
    - （）
    - （）

：
    1：——CartPole ""
    2：——
    3：——

：
    python debug_reward_hacking.py
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

# ，
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# ：（Q、、）
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
        """：， Q """
        return self.net(x)


class ReplayBuffer:
    """
    ：
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


class DQNAgent:
    """
    DQN ： Q 、、 ε-

    ，，
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
        """ Q """
        if len(self.buffer) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

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
        """ Q （）"""
        self.target_net.load_state_dict(self.q_net.state_dict())


# ==========================================
# ：（）
# ==========================================
class HackedRewardWrapper(gym.Wrapper):
    """
     —— ""

     CartPole ： +1，。
    ：。

    ""，，
    ：，
    。

    ：
    ""（），
    ，。
    """

    def __init__(self, env, survival_bonus=2.0):
        super().__init__(env)
        self.survival_bonus = survival_bonus

    def step(self, action):
        """
        ：
        - ： +1
        - ：""，
        - ， bonus → ""
        """
        next_state, reward, done, truncated, info = self.env.step(action)

        # ：，
        # "，"
        hacked_reward = reward + self.survival_bonus

        return next_state, hacked_reward, done, truncated, info


class BadShapingWrapper(gym.Wrapper):
    """
     —— ""

    ：。
    ""。
    ：，
    ！

    ：""""。
    """

    def __init__(self, env, position_reward_weight=5.0):
        super().__init__(env)
        self.position_reward_weight = position_reward_weight

    def step(self, action):
        """
        ：
        - ： +1（）
        - ：""

         position_reward_weight ，
        ""，""。
        ：，。
        """
        next_state, reward, done, truncated, info = self.env.step(action)

        # next_state[0] ， [-2.4, 2.4]
        # ""
        position = next_state[0]
        position_reward = self.position_reward_weight * (1.0 - abs(position) / 2.4)

        shaped_reward = reward + position_reward

        return next_state, shaped_reward, done, truncated, info


# ==========================================
# ：
# ==========================================
def train_agent(env, label, num_episodes=300, batch_size=64):
    """
     DQN ，

    ：
        env: （）
        label: （）
        num_episodes: 
        batch_size: 
    ：
        reward_history: （）
        task_score_history: （）
        loss_history: 
    """
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim, lr=1e-3, gamma=0.99)

    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    TARGET_UPDATE_FREQ = 10

    epsilon = EPSILON_START
    reward_history = []
    task_score_history = []  # （）
    loss_history = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_task_score = 0.0  #  CartPole 
        episode_loss = 0.0
        steps = 0

        while True:
            action = agent.select_action(state, epsilon)

            # （）
            next_state, reward, done, truncated, _ = env.step(action)

            # （）
            #  CartPole  1.0
            original_reward = 1.0

            agent.buffer.push(state, action, reward, next_state, float(done))
            loss = agent.update(batch_size)

            state = next_state
            episode_reward += reward
            episode_task_score += original_reward
            episode_loss += loss
            steps += 1

            if done or truncated:
                break

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        reward_history.append(episode_reward)
        task_score_history.append(episode_task_score)
        loss_history.append(episode_loss / max(steps, 1))

        if (episode + 1) % TARGET_UPDATE_FREQ == 0:
            agent.update_target()

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(reward_history[-50:])
            avg_task = np.mean(task_score_history[-50:])
            print(
                f"  [{label}]  {episode + 1:4d}/{num_episodes} | "
                f"(): {avg_reward:7.1f} | "
                f"(): {avg_task:5.1f}"
            )

    env.close()
    return reward_history, task_score_history, loss_history


# ==========================================
# ： —— 
# ==========================================
def diagnose_reward_hacking(reward_history, task_score_history, label):
    """
    

    ：
        - "，"
        - 
        -  → 
        -  → ！

    ：
        reward_history: 
        task_score_history: 
        label: 
    ：
        is_hacked: （True/False）
    """
    print(f"\n{'=' * 60}")
    print(f"   —— {label}")
    print(f"{'=' * 60}")

    #  1/4  1/4 
    n = len(reward_history)
    quarter = max(n // 4, 1)

    # 
    early_reward = np.mean(reward_history[:quarter])
    late_reward = np.mean(reward_history[-quarter:])
    reward_trend = late_reward - early_reward

    # 
    early_task = np.mean(task_score_history[:quarter])
    late_task = np.mean(task_score_history[-quarter:])
    task_trend = late_task - early_task

    print(f"  （）: {early_reward:.1f}")
    print(f"  （）: {late_reward:.1f}")
    print(f"  : {'↑' if reward_trend > 0 else '↓'} {abs(reward_trend):.1f}")
    print(f"  ─────────────────────────────")
    print(f"  （）: {early_task:.1f}")
    print(f"  （）: {late_task:.1f}")
    print(f"  : {'↑' if task_trend > 0 else '↓'} {abs(task_trend):.1f}")

    # ： = 
    is_hacked = (reward_trend > 0) and (task_trend < 0)

    # 
    if n > 10:
        correlation = np.corrcoef(reward_history, task_score_history)[0, 1]
        print(f"  : {correlation:.3f}")
    else:
        correlation = 1.0

    print(f"  ─────────────────────────────")
    if is_hacked:
        print(f"  ⚠️  ！")
        print(f"  ，。")
        print(f"  。")
        print(f"  ：。")
    elif correlation < 0.3:
        print(f"  ⚠️  （{correlation:.3f}）")
        print(f"  。")
        print(f"  ：。")
    else:
        print(f"  ✓  ，。")

    print(f"{'=' * 60}")

    return is_hacked


# ==========================================
# ：
# ==========================================
def run_all_experiments():
    """
    

    1：（HackedRewardWrapper）
    2：（BadShapingWrapper）
    3：（ CartPole ）
    """

    NUM_EPISODES = 300  #  300 

    print("=" * 60)
    print("  A：（Reward Hacking）")
    print("=" * 60)
    print(f"   {NUM_EPISODES} ")
    print(f"  ''''")
    print("=" * 60)

    # ── 1： ──
    print("\n" + "─" * 60)
    print("  1：")
    print("  ：，")
    print("  ：，")
    print("─" * 60)

    hacked_env = HackedRewardWrapper(
        gym.make("CartPole-v1"),
        survival_bonus=2.0,  # 
    )
    hacked_rewards, hacked_task_scores, hacked_losses = train_agent(
        hacked_env, "", num_episodes=NUM_EPISODES
    )
    diagnose_reward_hacking(hacked_rewards, hacked_task_scores, "1-")

    # ── 2： ──
    print("\n" + "─" * 60)
    print("  2：")
    print("  ：''，''")
    print("  ：，")
    print("─" * 60)

    bad_shaping_env = BadShapingWrapper(
        gym.make("CartPole-v1"),
        position_reward_weight=5.0,  # 
    )
    shaping_rewards, shaping_task_scores, shaping_losses = train_agent(
        bad_shaping_env, "", num_episodes=NUM_EPISODES
    )
    diagnose_reward_hacking(shaping_rewards, shaping_task_scores, "2-")

    # ── 3： ──
    print("\n" + "─" * 60)
    print("  3：（）")
    print("   CartPole ： +1，")
    print("  、")
    print("─" * 60)

    correct_env = gym.make("CartPole-v1")
    correct_rewards, correct_task_scores, correct_losses = train_agent(
        correct_env, "", num_episodes=NUM_EPISODES
    )
    diagnose_reward_hacking(correct_rewards, correct_task_scores, "3-")

    return (hacked_rewards, hacked_task_scores,
            shaping_rewards, shaping_task_scores,
            correct_rewards, correct_task_scores)


# ==========================================
# ：
# ==========================================
def plot_comparison(hacked_rewards, hacked_task_scores,
                    shaping_rewards, shaping_task_scores,
                    correct_rewards, correct_task_scores):
    """
    

    ：（）
    ：（）

    ""。
    """

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("： vs ",
                 fontsize=16, fontweight='bold')

    scenarios = [
        ("1：", hacked_rewards, hacked_task_scores, "#e74c3c"),
        ("2：", shaping_rewards, shaping_task_scores, "#e67e22"),
        ("3：", correct_rewards, correct_task_scores, "#27ae60"),
    ]

    window = 20  # 

    for col, (title, rewards, task_scores, color) in enumerate(scenarios):
        # ── ： ──
        ax_top = axes[0, col]
        ax_top.plot(rewards, alpha=0.2, color=color)
        if len(rewards) >= window:
            moving_avg = [
                np.mean(rewards[max(0, i - window): i + 1])
                for i in range(len(rewards))
            ]
            ax_top.plot(moving_avg, color=color, linewidth=2, label='')
        ax_top.set_title(f"{title}\n", fontsize=12)
        ax_top.set_xlabel('')
        ax_top.set_ylabel('（）')
        ax_top.grid(True, alpha=0.3)
        ax_top.legend(fontsize=9)

        # ── ： ──
        ax_bot = axes[1, col]
        ax_bot.plot(task_scores, alpha=0.2, color=color)
        if len(task_scores) >= window:
            moving_avg_task = [
                np.mean(task_scores[max(0, i - window): i + 1])
                for i in range(len(task_scores))
            ]
            ax_bot.plot(moving_avg_task, color=color, linewidth=2, label='')
        ax_bot.set_title(f"{title}\n", fontsize=12)
        ax_bot.set_xlabel('')
        ax_bot.set_ylabel('（）')
        ax_bot.grid(True, alpha=0.3)
        ax_bot.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("output/reward_hacking_diagnosis.png", dpi=150, bbox_inches='tight')
    print("\n output/reward_hacking_diagnosis.png")
    plt.show()


# ==========================================
# ：
# ==========================================
def print_reward_design_principles():
    """
    

    ，
     OpenAI  DeepMind 。
    """
    print("\n" + "=" * 60)
    print("  （）")
    print("=" * 60)

    principles = [
        ("1. ",
         "。CartPole  +1，"
         "，。"),

        ("2. ",
         "。''，"
         "。"),

        ("3. ",
         "（Reward Shaping）。"
         "，。"
         "，（Potential-Based Shaping），"
         "。"),

        ("4. ",
         "。：、、。"
         "，。"),

        ("5. ，",
         "，。"
         "，。"),

        ("6. ",
         "，。"
         "，"
         "''。"),
    ]

    for title, description in principles:
        print(f"\n  {title}")
        print(f"    {description}")

    print("\n" + "=" * 60)
    print("  ：")
    print("=" * 60)

    checklist = [
        "□ ？",
        "□ （）？",
        "□  0.5？",
        "□ （）？",
        "□ ，？",
        "□ ''？",
    ]

    for item in checklist:
        print(f"  {item}")

    print("=" * 60)


# ==========================================
# 
# ==========================================
if __name__ == "__main__":
    # 
    results = run_all_experiments()

    # 
    (hacked_rewards, hacked_task_scores,
     shaping_rewards, shaping_task_scores,
     correct_rewards, correct_task_scores) = results

    # 
    plot_comparison(
        hacked_rewards, hacked_task_scores,
        shaping_rewards, shaping_task_scores,
        correct_rewards, correct_task_scores,
    )

    # 
    print_reward_design_principles()
