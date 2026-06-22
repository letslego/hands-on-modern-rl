"""
13： (Multi-Agent RL) 
——

：
- 3
- ，
- ""（）
- ：
    1.  (Independent Q-Learning)：
    2.  (Shared Policy)： Q 

：
    python multi_agent_marl.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 
os.makedirs("output", exist_ok=True)

# 
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# ：
# ==========================================
class MultiAgentGridWorld:
    """
    

    （8×8）：
        - 0: 
        - 1: （，）
        - 2: （）

    ：
        - 3，
        - ：、、、、
        - 

    ：
        - ： +3 
        - 

    ：
        - : +5
        - （）:  +3
        - : -0.5（）
        - : -1
    """

    # ：、、、、
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
    ACTION_NAMES = ['', '', '', '', '']
    N_ACTIONS = 5

    def __init__(self, grid_size=8, n_agents=3, n_resources=6):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.n_resources = n_resources

        # 
        self.walls = set()
        # ，
        wall_positions = [(3, 3), (3, 4), (4, 3)]
        for wp in wall_positions:
            if 0 <= wp[0] < grid_size and 0 <= wp[1] < grid_size:
                self.walls.add(wp)

        # （）
        self.agent_starts = [(0, 0), (0, grid_size - 1), (grid_size - 1, 0)]

        # （）
        self.resource_positions = set()

        # 
        self.agent_positions = []

    def reset(self):
        """
        

        ，。
        """
        # 
        self.agent_positions = [list(pos) for pos in self.agent_starts[:self.n_agents]]

        # 
        self.resource_positions = set()
        while len(self.resource_positions) < self.n_resources:
            r = np.random.randint(0, self.grid_size)
            c = np.random.randint(0, self.grid_size)
            pos = (r, c)
            # 
            if pos not in self.walls and pos not in self.agent_starts:
                self.resource_positions.add(pos)

        return self._get_observations()

    def _get_observations(self):
        """
        

        ：
            - 
            - 
            - 

         Q-learning， ID。
        """
        obs = []
        for i in range(self.n_agents):
            r, c = self.agent_positions[i]
            obs.append((r, c))
        return obs

    def step(self, actions):
        """
        

        ：
            actions:  n_agents 

        ：
            observations: 
            rewards: 
            done: 
            info: 
        """
        total_resources_before = len(self.resource_positions)
        rewards = [0.0] * self.n_agents
        collected_positions = []

        # 
        for i in range(self.n_agents):
            action = actions[i]
            dr, dc = self.ACTIONS[action]

            new_r = self.agent_positions[i][0] + dr
            new_c = self.agent_positions[i][1] + dc

            # 
            if (new_r < 0 or new_r >= self.grid_size
                    or new_c < 0 or new_c >= self.grid_size
                    or (new_r, new_c) in self.walls):
                # ：，
                rewards[i] -= 1.0
                continue

            # 
            self.agent_positions[i] = [new_r, new_c]

            # 
            if action < 4:  # ""
                rewards[i] -= 0.5

            # 
            pos = (new_r, new_c)
            if pos in self.resource_positions:
                rewards[i] += 5.0
                collected_positions.append((i, pos))
                self.resource_positions.discard(pos)

        # ：
        # 
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                dist = abs(self.agent_positions[i][0] - self.agent_positions[j][0]) + \
                       abs(self.agent_positions[i][1] - self.agent_positions[j][1])
                # ：（）
                if dist == 1:
                    # 
                    for _, res_pos in collected_positions:
                        dist_i = abs(self.agent_positions[i][0] - res_pos[0]) + \
                                 abs(self.agent_positions[i][1] - res_pos[1])
                        dist_j = abs(self.agent_positions[j][0] - res_pos[0]) + \
                                 abs(self.agent_positions[j][1] - res_pos[1])
                        if dist_i <= 1 and dist_j <= 1:
                            rewards[i] += 1.5  # 
                            rewards[j] += 1.5

        # 
        done = len(self.resource_positions) == 0

        # ，
        if done:
            for i in range(self.n_agents):
                rewards[i] += 3.0  # 

        info = {
            'resources_remaining': len(self.resource_positions),
            'resources_collected': total_resources_before - len(self.resource_positions),
        }

        return self._get_observations(), rewards, done, info


# ==========================================
# ：Q-Learning 
# ==========================================
class QLearningAgent:
    """
    Q-Learning 

     ε- Q-learning 。
     Q 。
    """

    def __init__(self, agent_id, n_actions=5, lr=0.1, gamma=0.95,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
        self.agent_id = agent_id
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q ：， (state, action)
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def get_state_key(self, obs, env):
        """
         Q 

        ：
            (, , , )

         + 。
        """
        r, c = obs
        resources_left = len(env.resource_positions)
        return (r, c, resources_left)

    def select_action(self, state_key):
        """ε-"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return int(np.argmax(self.q_table[state_key]))

    def update(self, state_key, action, reward, next_state_key, done):
        """
        Q-Learning 

        Q(s, a) ← Q(s, a) + α * [r + γ * max Q(s', a') - Q(s, a)]
        """
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state_key])

        td_error = target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.lr * td_error

    def decay_epsilon(self):
        """"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


class SharedPolicyAgent:
    """
    

     Q 。
    （ IPPO ）。

    ：
        - （3）
        - 

    ：
        - 
    """

    def __init__(self, n_agents=3, n_actions=5, lr=0.1, gamma=0.95,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        #  Q 
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def get_state_key(self, obs, env, agent_id):
        """
        

         agent_id ，
         Q 。
        """
        r, c = obs
        resources_left = len(env.resource_positions)
        return (agent_id, r, c, resources_left)

    def select_action(self, state_key):
        """ε-"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return int(np.argmax(self.q_table[state_key]))

    def update(self, state_key, action, reward, next_state_key, done):
        """Q-Learning （ Q ）"""
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state_key])

        td_error = target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.lr * td_error

    def decay_epsilon(self):
        """"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


# ==========================================
# ：
# ==========================================
def train_independent(env, n_episodes=800, max_steps=100, verbose=True):
    """
    （Independent Q-Learning, IQL）

     Q ，
    （）。

    ，。
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  ： (Independent Q-Learning)")
        print(f"  : {env.n_agents}")
        print(f"  : {n_episodes}")
        print(f"{'='*60}")

    agents = [QLearningAgent(agent_id=i) for i in range(env.n_agents)]

    # 
    episode_rewards = []       # 
    agent_rewards = [[] for _ in range(env.n_agents)]  # 
    cooperation_counts = []    # 
    completion_rates = []      # 

    for episode in range(n_episodes):
        obs = env.reset()
        total_team_reward = 0
        individual_rewards = [0.0] * env.n_agents
        cooperation_events = 0

        # 
        state_keys = [agents[i].get_state_key(obs[i], env) for i in range(env.n_agents)]

        for step in range(max_steps):
            # 
            actions = [agents[i].select_action(state_keys[i]) for i in range(env.n_agents)]

            # 
            next_obs, rewards, done, info = env.step(actions)

            # 
            next_state_keys = [agents[i].get_state_key(next_obs[i], env)
                               for i in range(env.n_agents)]

            #  Q 
            for i in range(env.n_agents):
                agents[i].update(state_keys[i], actions[i], rewards[i],
                                 next_state_keys[i], done)
                individual_rewards[i] += rewards[i]

            # 
            total_team_reward += sum(rewards)
            # ：
            positive_count = sum(1 for r in rewards if r > 2.0)
            if positive_count >= 2:
                cooperation_events += 1

            state_keys = next_state_keys

            if done:
                break

        # 
        for agent in agents:
            agent.decay_epsilon()

        # 
        episode_rewards.append(total_team_reward)
        for i in range(env.n_agents):
            agent_rewards[i].append(individual_rewards[i])
        cooperation_counts.append(cooperation_events)
        completion_rates.append(1.0 if done else 0.0)

        #  200 
        if verbose and (episode + 1) % 200 == 0:
            avg_reward = np.mean(episode_rewards[-200:])
            avg_coop = np.mean(cooperation_counts[-200:])
            avg_complete = np.mean(completion_rates[-200:])
            print(f"   {episode+1:4d} | "
                  f": {avg_reward:7.2f} | "
                  f": {avg_coop:4.1f} | "
                  f": {avg_complete:.0%} | "
                  f"ε: {agents[0].epsilon:.3f}")

    return {
        'episode_rewards': episode_rewards,
        'agent_rewards': agent_rewards,
        'cooperation_counts': cooperation_counts,
        'completion_rates': completion_rates,
        'agents': agents,
    }


def train_shared(env, n_episodes=800, max_steps=100, verbose=True):
    """
    

     Q 。
    （weight sharing），。
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  ： (Shared Policy)")
        print(f"  : {env.n_agents}")
        print(f"  : {n_episodes}")
        print(f"{'='*60}")

    shared_agent = SharedPolicyAgent(n_agents=env.n_agents)

    # 
    episode_rewards = []
    agent_rewards = [[] for _ in range(env.n_agents)]
    cooperation_counts = []
    completion_rates = []

    for episode in range(n_episodes):
        obs = env.reset()
        total_team_reward = 0
        individual_rewards = [0.0] * env.n_agents
        cooperation_events = 0

        state_keys = [shared_agent.get_state_key(obs[i], env, i)
                      for i in range(env.n_agents)]

        for step in range(max_steps):
            #  Q 
            actions = [shared_agent.select_action(state_keys[i])
                       for i in range(env.n_agents)]

            next_obs, rewards, done, info = env.step(actions)
            next_state_keys = [shared_agent.get_state_key(next_obs[i], env, i)
                               for i in range(env.n_agents)]

            #  Q （3）
            for i in range(env.n_agents):
                shared_agent.update(state_keys[i], actions[i], rewards[i],
                                    next_state_keys[i], done)
                individual_rewards[i] += rewards[i]

            total_team_reward += sum(rewards)
            positive_count = sum(1 for r in rewards if r > 2.0)
            if positive_count >= 2:
                cooperation_events += 1

            state_keys = next_state_keys

            if done:
                break

        shared_agent.decay_epsilon()

        episode_rewards.append(total_team_reward)
        for i in range(env.n_agents):
            agent_rewards[i].append(individual_rewards[i])
        cooperation_counts.append(cooperation_events)
        completion_rates.append(1.0 if done else 0.0)

        if verbose and (episode + 1) % 200 == 0:
            avg_reward = np.mean(episode_rewards[-200:])
            avg_coop = np.mean(cooperation_counts[-200:])
            avg_complete = np.mean(completion_rates[-200:])
            print(f"   {episode+1:4d} | "
                  f": {avg_reward:7.2f} | "
                  f": {avg_coop:4.1f} | "
                  f": {avg_complete:.0%} | "
                  f"ε: {shared_agent.epsilon:.3f}")

    return {
        'episode_rewards': episode_rewards,
        'agent_rewards': agent_rewards,
        'cooperation_counts': cooperation_counts,
        'completion_rates': completion_rates,
        'agent': shared_agent,
    }


# ==========================================
# ：
# ==========================================
def visualize_results(ind_results, shared_results, n_agents=3):
    """
    

    4：
        1. 
        2. 
        3. 
        4. 
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(' —  vs ',
                 fontsize=16, fontweight='bold')

    window = 50  # 
    episodes = range(len(ind_results['episode_rewards']))

    # ---- 1： ----
    ax1 = axes[0, 0]

    # 
    ind_rewards = ind_results['episode_rewards']
    if len(ind_rewards) >= window:
        ind_avg = np.convolve(ind_rewards, np.ones(window) / window, mode='valid')
        ax1.plot(range(window - 1, len(ind_rewards)), ind_avg,
                 color='#F44336', linewidth=2.5, label=' (IQL)')

    # 
    shared_rewards = shared_results['episode_rewards']
    if len(shared_rewards) >= window:
        shared_avg = np.convolve(shared_rewards, np.ones(window) / window, mode='valid')
        ax1.plot(range(window - 1, len(shared_rewards)), shared_avg,
                 color='#2196F3', linewidth=2.5, label=' (Shared)')

    ax1.set_xlabel('', fontsize=12)
    ax1.set_ylabel('', fontsize=12)
    ax1.set_title('', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # ---- 2： ----
    ax2 = axes[0, 1]

    ind_completion = ind_results['completion_rates']
    shared_completion = shared_results['completion_rates']

    # 
    if len(ind_completion) >= window:
        ind_comp_avg = np.convolve(ind_completion, np.ones(window) / window, mode='valid')
        ax2.plot(range(window - 1, len(ind_completion)), ind_comp_avg * 100,
                 color='#F44336', linewidth=2.5, label=' (IQL)')

    if len(shared_completion) >= window:
        shared_comp_avg = np.convolve(shared_completion, np.ones(window) / window, mode='valid')
        ax2.plot(range(window - 1, len(shared_completion)), shared_comp_avg * 100,
                 color='#2196F3', linewidth=2.5, label=' (Shared)')

    ax2.set_xlabel('', fontsize=12)
    ax2.set_ylabel(' (%)', fontsize=12)
    ax2.set_title('', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)

    # ---- 3： ----
    ax3 = axes[1, 0]

    ind_coop = ind_results['cooperation_counts']
    shared_coop = shared_results['cooperation_counts']

    if len(ind_coop) >= window:
        ind_coop_avg = np.convolve(ind_coop, np.ones(window) / window, mode='valid')
        ax3.plot(range(window - 1, len(ind_coop)), ind_coop_avg,
                 color='#F44336', linewidth=2.5, label=' (IQL)')

    if len(shared_coop) >= window:
        shared_coop_avg = np.convolve(shared_coop, np.ones(window) / window, mode='valid')
        ax3.plot(range(window - 1, len(shared_coop)), shared_coop_avg,
                 color='#2196F3', linewidth=2.5, label=' (Shared)')

    ax3.set_xlabel('', fontsize=12)
    ax3.set_ylabel('', fontsize=12)
    ax3.set_title('', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)

    # ---- 4：（） ----
    ax4 = axes[1, 1]

    agent_colors = ['#4CAF50', '#FF9800', '#9C27B0']
    for i in range(n_agents):
        rewards_i = shared_results['agent_rewards'][i]
        if len(rewards_i) >= window:
            avg_i = np.convolve(rewards_i, np.ones(window) / window, mode='valid')
            ax4.plot(range(window - 1, len(rewards_i)), avg_i,
                     color=agent_colors[i], linewidth=2, label=f' {i+1}')

    ax4.set_xlabel('', fontsize=12)
    ax4.set_ylabel('', fontsize=12)
    ax4.set_title('（）', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def visualize_trajectories(env, shared_agent, max_steps=30):
    """
    

    3。
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('', fontsize=16, fontweight='bold')

    # 
    obs = env.reset()
    agent_colors = ['#4CAF50', '#FF9800', '#9C27B0']
    trajectories = [[tuple(obs[i])] for i in range(env.n_agents)]
    initial_resources = set(env.resource_positions)
    collected_steps = []  # 

    for step in range(max_steps):
        state_keys = [shared_agent.get_state_key(obs[i], env, i)
                      for i in range(env.n_agents)]
        actions = [shared_agent.select_action(state_keys[i])
                   for i in range(env.n_agents)]
        next_obs, rewards, done, info = env.step(actions)

        for i in range(env.n_agents):
            trajectories[i].append(tuple(next_obs[i]))

        if info['resources_collected'] > 0:
            collected_steps.append(step)

        obs = next_obs
        if done:
            break

    # ---- ： ----
    ax1 = axes[0]

    # 
    grid_display = np.zeros((env.grid_size, env.grid_size))
    for wr, wc in env.walls:
        grid_display[wr][wc] = -1

    ax1.imshow(grid_display, cmap='Greys', alpha=0.3,
               extent=[-0.5, env.grid_size - 0.5, env.grid_size - 0.5, -0.5])

    # 
    for rr, rc in initial_resources:
        ax1.plot(rc, rr, '*', color='gold', markersize=15,
                 markeredgecolor='orange', markeredgewidth=1.5)

    # 
    for i in range(env.n_agents):
        traj = trajectories[i]
        rows = [t[0] for t in traj]
        cols = [t[1] for t in traj]

        # 
        ax1.plot(cols, rows, '-', color=agent_colors[i], linewidth=2,
                 alpha=0.7, label=f' {i+1}')
        # 
        ax1.plot(cols[0], rows[0], 'o', color=agent_colors[i], markersize=12)
        # 
        ax1.plot(cols[-1], rows[-1], 's', color=agent_colors[i], markersize=12)

        # 
        for step_idx, (r, c) in enumerate(traj):
            if step_idx % 5 == 0 and step_idx > 0:  # 5
                ax1.text(c + 0.2, r - 0.2, str(step_idx), fontsize=7,
                         color=agent_colors[i], alpha=0.7)

    # 
    for wr, wc in env.walls:
        ax1.add_patch(plt.Rectangle((wc - 0.5, wr - 0.5), 1, 1,
                                     facecolor='gray', alpha=0.5))
        ax1.text(wc, wr, '', ha='center', va='center', fontsize=10,
                 color='white', fontweight='bold')

    ax1.set_xlim(-0.5, env.grid_size - 0.5)
    ax1.set_ylim(env.grid_size - 0.5, -0.5)
    ax1.set_xticks(range(env.grid_size))
    ax1.set_yticks(range(env.grid_size))
    ax1.set_title('', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ---- ： vs  ----
    ax2 = axes[1]

    total_steps = len(trajectories[0])
    resources_per_step = []
    remaining = len(initial_resources)

    # 
    obs2 = env.reset()
    env.resource_positions = set(initial_resources)
    remaining_track = [len(env.resource_positions)]

    for step in range(total_steps - 1):
        state_keys = [shared_agent.get_state_key(obs2[i], env, i)
                      for i in range(env.n_agents)]
        actions = [shared_agent.select_action(state_keys[i])
                   for i in range(env.n_agents)]
        next_obs, rewards, done, info = env.step(actions)
        remaining_track.append(info['resources_remaining'])
        obs2 = next_obs
        if done:
            break

    ax2.fill_between(range(len(remaining_track)), remaining_track,
                     alpha=0.3, color='#4CAF50')
    ax2.plot(range(len(remaining_track)), remaining_track,
             '-o', color='#4CAF50', linewidth=2, markersize=5)
    ax2.set_xlabel('', fontsize=12)
    ax2.set_ylabel('', fontsize=12)
    ax2.set_title('', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 
    if remaining_track[-1] == 0:
        complete_step = len(remaining_track) - 1
        ax2.axvline(x=complete_step, color='red', linestyle='--', linewidth=1.5)
        ax2.annotate(f'\n({complete_step})',
                     xy=(complete_step, 0), xytext=(complete_step + 2, 2),
                     fontsize=10, color='red', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='red'))

    plt.tight_layout()
    return fig


def print_cooperation_statistics(ind_results, shared_results, n_agents):
    """
    
    """
    print("\n" + "=" * 60)
    print("  ")
    print("=" * 60)

    # 200
    last_n = 200

    for name, results in [("", ind_results), ("", shared_results)]:
        rewards = results['episode_rewards'][-last_n:]
        completion = results['completion_rates'][-last_n:]
        coop = results['cooperation_counts'][-last_n:]

        print(f"\n  [{name}]  {last_n} :")
        print(f"    :   {np.mean(rewards):.2f}")
        print(f"    :     {np.mean(completion):.0%}")
        print(f"    :   {np.mean(coop):.2f}")
        print(f"    : {np.std(rewards):.2f}")

        # 
        print(f"    :", end="")
        for i in range(n_agents):
            avg_r = np.mean(results['agent_rewards'][i][-last_n:])
            print(f"  A{i+1}={avg_r:.1f}", end="")
        print()

    print("\n" + "-" * 60)


# ==========================================
# ：
# ==========================================
def main():
    """
    ： →  →  → 

    ：
        1. 
        2.  (IQL) 
        3. 
        4. 、、
        5. 
    """

    print("=" * 60)
    print("  13： (MARL) ")
    print("=" * 60)
    print("  : ")
    print("  : 3 ( Q-Learning vs )")
    print("  : 8x86")
    print("-" * 60)

    # 
    np.random.seed(42)

    # ---- 1： ----
    print("\n[1] ")
    env = MultiAgentGridWorld(grid_size=8, n_agents=3, n_resources=6)
    obs = env.reset()
    print(f"  : {env.grid_size}×{env.grid_size}")
    print(f"  : {env.agent_starts}")
    print(f"  : {env.n_resources}")
    print(f"  : {env.walls}")

    # ---- 2： ----
    print("\n[2]  (IQL) ...")
    ind_results = train_independent(env, n_episodes=800, max_steps=80)

    # ---- 3： ----
    print("\n[3] ...")
    shared_results = train_shared(env, n_episodes=800, max_steps=80)

    # ---- 4： ----
    print("\n[4] ")
    print_cooperation_statistics(ind_results, shared_results, env.n_agents)

    # ---- 5： ----
    print("[5] ...")

    # 1：
    fig1 = visualize_results(ind_results, shared_results, env.n_agents)
    fig1.savefig('output/marl_training_comparison.png', dpi=150, bbox_inches='tight')
    print("  : output/marl_training_comparison.png")

    # 2：
    fig2 = visualize_trajectories(env, shared_results['agent'], max_steps=40)
    fig2.savefig('output/marl_trajectories.png', dpi=150, bbox_inches='tight')
    print("  : output/marl_trajectories.png")

    # ----  ----
    print("\n" + "=" * 60)
    print("  ")
    print("=" * 60)
    print("  1.  (IQL): ，")
    print("  2. : ，")
    print("  3.  RL ")
    print("  4. :")
    print("     -  ()")
    print("     -  ()")
    print("     -  ()")
    print("  5.  RL :")
    print("     -  (Self-Play) ")
    print("     - ")
    print("     - ")
    print("=" * 60)

    plt.show()


if __name__ == "__main__":
    main()
