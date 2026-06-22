"""
A - ：（Training Collapse）

：。

？
    、、。
    ""，，
    。

：
    -  NaN 
    - 
    - 
    - （）

：
    1： → 
    2： → 
    3：ε  → 

，4。

：
    python debug_training_collapse.py
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
# ：（Q、）
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
class ConfigurableDQNAgent:
    """
     DQN  —— 

    ：
        lr: （）
        clip_grad: 
        clip_max_norm: 
        epsilon_start: 
        epsilon_end: 
        epsilon_decay: （ 1.0 ）
    """

    def __init__(self, state_dim, action_dim,
                 lr=1e-3, gamma=0.99,
                 clip_grad=True, clip_max_norm=10.0,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.action_dim = action_dim
        self.gamma = gamma

        # Q 
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # 
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # 
        self.buffer = ReplayBuffer(capacity=10000)

        # 
        self.clip_grad = clip_grad
        self.clip_max_norm = clip_max_norm

        # 
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # ：
        self.last_grad_norm = 0.0

    def select_action(self, state):
        """ε-"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def decay_epsilon(self):
        """"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update(self, batch_size):
        """
         Q 
        ：loss （ NaN ）
        """
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

        # （）
        total_norm = 0.0
        for p in self.q_net.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        self.last_grad_norm = total_norm

        # 
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self.q_net.parameters(), max_norm=self.clip_max_norm
            )

        self.optimizer.step()

        return loss.item()

    def update_target(self):
        """ Q """
        self.target_net.load_state_dict(self.q_net.state_dict())


# ==========================================
# ：
# ==========================================
def train_scenario(agent, env, label, num_episodes=300, batch_size=64,
                   target_update_freq=10, verbose=True):
    """
     DQN 

    ，

    ：
        agent:  DQN 
        env: 
        label: 
        num_episodes: 
        batch_size: 
        target_update_freq: 
        verbose: 
    ：
        results: ，
    """
    # 
    reward_history = []
    loss_history = []
    grad_norm_history = []
    action_counts = {0: [], 1: []}  # 
    nan_detected = False

    if verbose:
        print(f"\n{'─' * 60}")
        print(f"  ：{label}")
        print(f"{'─' * 60}")

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_loss = 0.0
        episode_grad_norm = 0.0
        steps = 0
        actions_taken = {0: 0, 1: 0}

        while True:
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            agent.buffer.push(state, action, reward, next_state, float(done))
            loss = agent.update(batch_size)

            #  NaN（）
            if np.isnan(loss) or np.isinf(loss):
                nan_detected = True
                if verbose and episode < num_episodes - 1:
                    print(f"  [!]  {episode + 1}  NaN/Inf！loss={loss}")
                loss = 0.0  #  0 

            state = next_state
            episode_reward += reward
            episode_loss += loss
            episode_grad_norm += agent.last_grad_norm
            actions_taken[action] = actions_taken.get(action, 0) + 1
            steps += 1

            if done or truncated:
                break

        # 
        agent.decay_epsilon()

        # 
        reward_history.append(episode_reward)
        loss_history.append(episode_loss / max(steps, 1))
        grad_norm_history.append(episode_grad_norm / max(steps, 1))
        action_counts[0].append(actions_taken.get(0, 0))
        action_counts[1].append(actions_taken.get(1, 0))

        # 
        if (episode + 1) % target_update_freq == 0:
            agent.update_target()

        # 
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(reward_history[-50:])
            avg_loss = np.mean(loss_history[-50:])
            avg_grad = np.mean(grad_norm_history[-50:])
            print(
                f"   {episode + 1:4d}/{num_episodes} | "
                f": {avg_reward:6.1f} | "
                f": {avg_loss:8.4f} | "
                f": {avg_grad:8.2f} | "
                f"ε: {agent.epsilon:.3f}"
            )

    env.close()

    return {
        'label': label,
        'rewards': reward_history,
        'losses': loss_history,
        'grad_norms': grad_norm_history,
        'action_counts': action_counts,
        'nan_detected': nan_detected,
    }


# ==========================================
# ：4
# ==========================================
def four_step_diagnosis(results):
    """
    4：

    1：（/？）
    2：（？）
    3：（？）
    4：（？）
    """
    label = results['label']
    losses = results['losses']
    rewards = results['rewards']
    action_counts = results['action_counts']
    grad_norms = results['grad_norms']

    print(f"\n{'=' * 60}")
    print(f"  4 —— {label}")
    print(f"{'=' * 60}")

    # ── 1： ──
    print(f"\n  1：")
    print(f"  {'─' * 40}")

    #  NaN/Inf 
    valid_losses = [l for l in losses if not np.isnan(l) and not np.isinf(l)]

    if len(valid_losses) == 0:
        print(f"     NaN/Inf → ！")
        print(f"    ：")
        print(f"    ： 1e-3 ")
    else:
        avg_loss = np.mean(valid_losses)
        max_loss = np.max(valid_losses)
        min_loss = np.min(valid_losses)

        print(f"    : [{min_loss:.4f}, {max_loss:.4f}]")
        print(f"    : {avg_loss:.4f}")

        if max_loss > 1000:
            print(f"    ⚠️  ！ {max_loss:.1f} ")
            print(f"    ：，")
        elif max_loss > 100:
            print(f"    ⚠️  ，")
        else:
            print(f"    ✓  ")

    # ── 2： ──
    print(f"\n  2：")
    print(f"  {'─' * 40}")

    n = len(rewards)
    quarter = max(n // 4, 1)
    early_reward = np.mean(rewards[:quarter])
    late_reward = np.mean(rewards[-quarter:])
    reward_change = late_reward - early_reward

    print(f"    : {early_reward:.1f}")
    print(f"    : {late_reward:.1f}")
    print(f"    : {'↑' if reward_change > 0 else '↓'} {abs(reward_change):.1f}")

    if reward_change > 10:
        print(f"    ✓  ，")
    elif reward_change > 0:
        print(f"    ~  ，")
    else:
        print(f"    ⚠️  ，")
        print(f"    ：、、")

    # ── 3： ──
    print(f"\n  3：")
    print(f"  {'─' * 40}")

    #  50 
    last_n = min(50, n)
    left_actions = np.sum(action_counts[0][-last_n:])
    right_actions = np.sum(action_counts[1][-last_n:])
    total_actions = left_actions + right_actions

    if total_actions > 0:
        left_ratio = left_actions / total_actions
        right_ratio = right_actions / total_actions
        print(f"     {last_n} :")
        print(f"      : {left_ratio:.1%} ({left_actions} )")
        print(f"      : {right_ratio:.1%} ({right_actions} )")

        if left_ratio > 0.95 or right_ratio > 0.95:
            print(f"    ⚠️  ！（）")
            print(f"    ：ε ")
        elif left_ratio > 0.8 or right_ratio > 0.8:
            print(f"    ~  ，")
        else:
            print(f"    ✓  ，")
    else:
        print(f"    ")

    # ── 4： ──
    print(f"\n  4：")
    print(f"  {'─' * 40}")

    valid_grads = [g for g in grad_norms if not np.isnan(g) and not np.isinf(g)]

    if len(valid_grads) == 0:
        print(f"     NaN/Inf → ！")
    else:
        avg_grad = np.mean(valid_grads)
        max_grad = np.max(valid_grads)

        print(f"    : {avg_grad:.2f}")
        print(f"    : {max_grad:.2f}")

        if max_grad > 1000:
            print(f"    ⚠️  ！ {max_grad:.1f}")
            print(f"    ： torch.nn.utils.clip_grad_norm_")
        elif max_grad > 100:
            print(f"    ~  ，")
        else:
            print(f"    ✓  ")

    print(f"{'=' * 60}")


# ==========================================
# ：
# ==========================================
def run_all_scenarios():
    """
    

    1：（lr=1.0）
    2：
    3：ε （ ε=1.0）
    4：
    """
    NUM_EPISODES = 300

    print("=" * 60)
    print("  A：")
    print("=" * 60)
    print(f"   {NUM_EPISODES} ")
    print(f"  3，4")
    print("=" * 60)

    # ── 1： ──
    print("\n" + "=" * 60)
    print("  1： (lr=1.0)")
    print("   1e-3， 1.0（1000）")
    print("  ：， NaN")
    print("=" * 60)

    env1 = gym.make("CartPole-v1")
    state_dim = env1.observation_space.shape[0]
    action_dim = env1.action_space.n

    agent_lr_high = ConfigurableDQNAgent(
        state_dim, action_dim,
        lr=1.0,               # ！ 1e-3
        clip_grad=True,        # 
        clip_max_norm=10.0,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
    )
    results_lr_high = train_scenario(agent_lr_high, env1, "lr=1.0（）")

    # ── 2： ──
    print("\n" + "=" * 60)
    print("  2：")
    print("  ，")
    print("  ：，")
    print("=" * 60)

    env2 = gym.make("CartPole-v1")
    agent_no_clip = ConfigurableDQNAgent(
        state_dim, action_dim,
        lr=1e-3,              # 
        clip_grad=False,       # ！
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
    )
    results_no_clip = train_scenario(agent_no_clip, env2, "")

    # ── 3：ε  ──
    print("\n" + "=" * 60)
    print("  3：ε （ ε=1.0）")
    print("  ，")
    print("  ：，")
    print("=" * 60)

    env3 = gym.make("CartPole-v1")
    agent_no_decay = ConfigurableDQNAgent(
        state_dim, action_dim,
        lr=1e-3,
        clip_grad=True,
        clip_max_norm=10.0,
        epsilon_start=1.0,
        epsilon_end=1.0,       # ε ！
        epsilon_decay=1.0,     #  1.0 = 
    )
    results_no_decay = train_scenario(agent_no_decay, env3, "ε=1.0（）")

    # ── 4： ──
    print("\n" + "=" * 60)
    print("  4：")
    print("   +  + ε ")
    print("  ：，")
    print("=" * 60)

    env4 = gym.make("CartPole-v1")
    agent_correct = ConfigurableDQNAgent(
        state_dim, action_dim,
        lr=1e-3,              # 
        clip_grad=True,        # 
        clip_max_norm=10.0,    # 
        epsilon_start=1.0,     #  ε
        epsilon_end=0.01,      #  ε
        epsilon_decay=0.995,   # 
    )
    results_correct = train_scenario(agent_correct, env4, "")

    return [results_lr_high, results_no_clip, results_no_decay, results_correct]


# ==========================================
# ：
# ==========================================
def plot_diagnosis(all_results):
    """
    4

    4 x 4：
    - 1：（？）
    - 2：（？）
    - 3：（？）
    - 4：（？）

    ，。
    """
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    fig.suptitle("：4",
                 fontsize=18, fontweight='bold', y=1.01)

    colors = ['#e74c3c', '#e67e22', '#9b59b6', '#27ae60']
    titles = [r['label'] for r in all_results]

    for col, (results, color, title) in enumerate(zip(all_results, colors, titles)):
        losses = results['losses']
        rewards = results['rewards']
        action_counts = results['action_counts']
        grad_norms = results['grad_norms']

        #  NaN/Inf： None 
        safe_losses = [l if not (np.isnan(l) or np.isinf(l)) else None for l in losses]
        safe_grads = [g if not (np.isnan(g) or np.isinf(g)) else None for g in grad_norms]

        # ── 1： ──
        ax1 = axes[0, col]
        # 
        valid_x = [i for i, l in enumerate(safe_losses) if l is not None]
        valid_y = [l for l in safe_losses if l is not None]
        if valid_y:
            ax1.plot(valid_x, valid_y, color=color, alpha=0.6, linewidth=0.8)
            # ，
            log_losses = [np.log10(max(l, 1e-8)) for l in valid_y]
            ax1.plot(valid_x, log_losses, color='navy', alpha=0.8, linewidth=1.5,
                     label='log10(loss)')
            ax1.legend(fontsize=8)
        ax1.set_title(f"{title}\n", fontsize=11)
        ax1.set_ylabel(' / log10()')
        ax1.grid(True, alpha=0.3)

        # ── 2： ──
        ax2 = axes[1, col]
        ax2.plot(rewards, color=color, alpha=0.3, linewidth=0.8)
        window = 20
        if len(rewards) >= window:
            moving_avg = [
                np.mean(rewards[max(0, i - window): i + 1])
                for i in range(len(rewards))
            ]
            ax2.plot(moving_avg, color='navy', linewidth=2, label='')
            ax2.legend(fontsize=8)
        ax2.set_title(f"", fontsize=11)
        ax2.set_ylabel('')
        ax2.grid(True, alpha=0.3)

        # ── 3： ──
        ax3 = axes[2, col]
        left_counts = np.array(action_counts[0], dtype=float)
        right_counts = np.array(action_counts[1], dtype=float)
        total_counts = left_counts + right_counts
        total_counts = np.where(total_counts == 0, 1, total_counts)  # 

        left_ratio = left_counts / total_counts
        right_ratio = right_counts / total_counts

        ax3.fill_between(range(len(left_ratio)), 0, left_ratio,
                         color='#3498db', alpha=0.6, label='')
        ax3.fill_between(range(len(right_ratio)), left_ratio, 1,
                         color='#e74c3c', alpha=0.6, label='')
        ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax3.set_title(f"", fontsize=11)
        ax3.set_ylabel('')
        ax3.set_ylim(0, 1)
        ax3.legend(fontsize=8, loc='upper right')
        ax3.grid(True, alpha=0.3)

        # ── 4： ──
        ax4 = axes[3, col]
        valid_gx = [i for i, g in enumerate(safe_grads) if g is not None]
        valid_gy = [g for g in safe_grads if g is not None]
        if valid_gy:
            ax4.plot(valid_gx, valid_gy, color=color, alpha=0.6, linewidth=0.8)
            # 
            log_grads = [np.log10(max(g, 1e-8)) for g in valid_gy]
            ax4.plot(valid_gx, log_grads, color='navy', alpha=0.8, linewidth=1.5,
                     label='log10(grad_norm)')
            ax4.axhline(y=np.log10(10), color='red', linestyle='--', alpha=0.5,
                        label='(10)')
            ax4.legend(fontsize=8)
        ax4.set_title(f"", fontsize=11)
        ax4.set_xlabel('')
        ax4.set_ylabel(' / log10()')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/training_collapse_diagnosis.png", dpi=150, bbox_inches='tight')
    print("\n output/training_collapse_diagnosis.png")
    plt.show()


# ==========================================
# ：
# ==========================================
def print_fix_summary():
    """
    
    """
    print("\n" + "=" * 60)
    print("  ")
    print("=" * 60)

    fixes = [
        {
            "": "（Loss  NaN/Inf）",
            "": "loss ",
            "": "",
            "": "（ 1e-4 ~ 1e-3），",
            "": "optimizer = Adam(params, lr=1e-3)  #  1.0",
        },
        {
            "": "（）",
            "": " > 1000 ",
            "": "",
            "": " clip_grad_norm_",
            "": "torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)",
        },
        {
            "": "（）",
            "": "",
            "": " ε ",
            "": " ε ",
            "": "epsilon = max(0.01, epsilon * 0.995)  # ",
        },
        {
            "": "（）",
            "": "（ 99:1）",
            "": "ε ",
            "": " ε ， ε ",
            "": "epsilon = max(0.05, epsilon * 0.999)  # ",
        },
    ]

    for i, fix in enumerate(fixes, 1):
        print(f"\n   {i}：{fix['']}")
        print(f"    ：{fix['']}")
        print(f"    ：{fix['']}")
        print(f"    ：{fix['']}")
        print(f"    ：{fix['']}")

    # 
    print(f"\n{'=' * 60}")
    print("  4")
    print(f"{'=' * 60}")

    steps = [
        ("1：",
         " loss 。",
         " loss > 1000  NaN → 。",
         "：，。"),
        ("2：",
         "。",
         " → 。",
         "：、、。"),
        ("3：",
         "。",
         " > 95% → 。",
         "： ε ，。"),
        ("4：",
         "。",
         " > 100 → 。",
         "：。"),
    ]

    for title, line1, line2, line3 in steps:
        print(f"\n  {title}")
        print(f"    {line1}")
        print(f"    {line2}")
        print(f"    {line3}")

    # 
    print(f"\n{'=' * 60}")
    print("  ")
    print(f"{'=' * 60}")

    checklist = [
        "□ （ NaN/Inf/）？",
        "□ （）？",
        "□ ？",
        "□ （）？",
        "□ （）？",
        "□ ε （）？",
        "□ （1e-4 ~ 1e-3）？",
        "□ （max_norm=10 ）？",
    ]

    for item in checklist:
        print(f"  {item}")

    print("=" * 60)


# ==========================================
# 
# ==========================================
if __name__ == "__main__":
    # 
    all_results = run_all_scenarios()

    # 4
    for results in all_results:
        four_step_diagnosis(results)

    # 
    plot_diagnosis(all_results)

    # 
    print_fix_summary()
