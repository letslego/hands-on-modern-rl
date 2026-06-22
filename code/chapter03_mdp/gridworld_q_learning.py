"""
3：4×4 GridWorld Q-Learning 
， Q 

：
    python gridworld_q_learning.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# 
os.makedirs("output", exist_ok=True)


# ==========================================
# ：GridWorld 
# ==========================================
class GridWorld:
    """
    4×4 

    （44）：
        0,0  0,1  0,2  0,3
        1,0  1,1  1,2  1,3
        2,0  2,1  2,2  2,3
        3,0  3,1  3,2  3,3

    - ：(0, 0)
    - ：(3, 3)， +10 
    - ：(1, 1)  (2, 2)， -5 
    - ：-1 （）
    - ：-5 （）

    ：
        0 =  (↑), 1 =  (↓), 2 =  (←), 3 =  (→)
    """

    def __init__(self):
        self.rows = 4
        self.cols = 4
        self.start = (0, 0)
        self.goal = (3, 3)
        self.obstacles = [(1, 1), (2, 2)]
        self.n_actions = 4  # 、、、
        self.action_names = ['(↑)', '(↓)', '(←)', '(→)']
        self.reset()

    def reset(self):
        """，"""
        self.agent_pos = self.start
        return self.agent_pos

    def step(self, action):
        """
        ， (, , )

        ：
            0 =  →  -1
            1 =  →  +1
            2 =  →  -1
            3 =  →  +1
        """
        row, col = self.agent_pos

        # 
        if action == 0:    # 
            new_pos = (row - 1, col)
        elif action == 1:  # 
            new_pos = (row + 1, col)
        elif action == 2:  # 
            new_pos = (row, col - 1)
        elif action == 3:  # 
            new_pos = (row, col + 1)
        else:
            raise ValueError(f": {action}")

        # （）
        new_row, new_col = new_pos
        if new_row < 0 or new_row >= self.rows or new_col < 0 or new_col >= self.cols:
            # ：，
            return self.agent_pos, -5, False

        # 
        if new_pos in self.obstacles:
            # ：，
            return self.agent_pos, -5, False

        # ：
        self.agent_pos = new_pos

        # 
        if self.agent_pos == self.goal:
            return self.agent_pos, 10, True  # ，+10 

        # ：-1 （）
        return self.agent_pos, -1, False


# ==========================================
# ：Q-Learning 
# ==========================================
def epsilon_greedy(Q, state, epsilon, n_actions):
    """
    ε-

     ε ， 1-ε  Q 。
     Q-Learning """"。
    """
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)  # ：
    else:
        return np.argmax(Q[state])  # ： Q 


def train_q_learning(env, n_episodes=500, alpha=0.1, gamma=0.95,
                     epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    """
    Q-Learning 

    Q-Learning （）：
        Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]

    ：
        - s: 
        - a: 
        - r: 
        - s': 
        - α: （）
        - γ: （）
        - max_a' Q(s', a'):  Q 

    ：
        n_episodes: 
        alpha: 
        gamma: 
        epsilon_start: 
        epsilon_end: 
        epsilon_decay: 
    """
    #  Q ： Q  0
    # Q[state][action] = 
    Q = np.zeros((env.rows, env.cols, env.n_actions))

    # 
    episode_rewards = []  # 
    episode_steps = []    # 
    epsilon = epsilon_start

    print("=" * 60)
    print("  Q-Learning ")
    print("=" * 60)
    print(f"   α = {alpha}")
    print(f"   γ = {gamma}")
    print(f"   ε = {epsilon_start}")
    print(f"   = {n_episodes}")
    print("-" * 60)

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            # 1.  ε-
            action = epsilon_greedy(Q, state, epsilon, env.n_actions)

            # 2. ，
            next_state, reward, done = env.step(action)

            # 3. Q-Learning （）
            #    ： max_a' Q(s', a')，
            #     Q-Learning "off-policy" 
            best_next_q = np.max(Q[next_state])
            td_target = reward + gamma * best_next_q
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error

            # 4. 
            state = next_state
            total_reward += reward
            steps += 1

            # ：
            if steps > 200:
                break

        # 
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        episode_rewards.append(total_reward)
        episode_steps.append(steps)

        #  100 
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_steps = np.mean(episode_steps[-100:])
            print(f"   {episode + 1:4d} | "
                  f"100: {avg_reward:7.2f} | "
                  f": {avg_steps:5.1f} | "
                  f"ε: {epsilon:.4f}")

    print("-" * 60)
    return Q, episode_rewards, episode_steps


# ==========================================
# ：
# ==========================================
def print_q_table(Q, env):
    """
     Q 

    Q  Q （）。
    Q ，。
    """
    print("\n" + "=" * 60)
    print("   Q ")
    print("=" * 60)
    print(f"{'':<10s}", end="")
    for name in env.action_names:
        print(f"{name:<12s}", end="")
    print(f"{'':<12s}")
    print("-" * 60)

    for r in range(env.rows):
        for c in range(env.cols):
            state = (r, c)
            if state in env.obstacles:
                print(f"({r},{c})   ", end="")
                print("    ---      ---      ---      ---     ")
                continue
            if state == env.goal:
                print(f"({r},{c})   ", end="")
                print("    ---      ---      ---      ---     ")
                continue

            print(f"({r},{c})       ", end="")
            for a in range(env.n_actions):
                print(f"{Q[r][c][a]:>8.2f}   ", end="")
            best_action = np.argmax(Q[r][c])
            print(f"  {env.action_names[best_action]}")

    print("-" * 60)


def extract_optimal_path(Q, env):
    """
     Q 

     Q ，。
     step() ，。
    """
    # ：0=, 1=, 2=, 3=
    deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    state = env.start
    path = [state]
    visited = set()

    while state != env.goal:
        if state in visited:
            break  # 
        visited.add(state)
        action = np.argmax(Q[state])
        dr, dc = deltas[action]
        new_state = (state[0] + dr, state[1] + dc)

        # （、）
        if (0 <= new_state[0] < env.rows and 0 <= new_state[1] < env.cols
                and new_state not in env.obstacles):
            state = new_state
        # ，（， visited ）
        path.append(state)
        if state == env.goal:
            break

    return path


def visualize_results(Q, episode_rewards, env):
    """
     Q-Learning 
    - 1： Q 
    - 2：
    - 3：
    """
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(16, 12))

    # ------------------------------------------
    # 1： Q 
    # ------------------------------------------
    action_names_short = ['(↑)', '(↓)', '(←)', '(→)']

    for i in range(4):
        ax = fig.add_subplot(2, 3, i + 1)
        q_values = Q[:, :, i]  #  Q 

        im = ax.imshow(q_values, cmap='RdYlGn', aspect='equal')
        #  Q 
        for r in range(env.rows):
            for c in range(env.cols):
                if (r, c) in env.obstacles:
                    ax.text(c, r, 'X', ha='center', va='center',
                            fontsize=14, fontweight='bold', color='black')
                elif (r, c) == env.goal:
                    ax.text(c, r, 'G', ha='center', va='center',
                            fontsize=14, fontweight='bold', color='blue')
                else:
                    ax.text(c, r, f'{q_values[r, c]:.1f}', ha='center', va='center',
                            fontsize=9)
        ax.set_title(f'Q(s, {action_names_short[i]})', fontsize=12)
        ax.set_xticks(range(env.cols))
        ax.set_yticks(range(env.rows))
        ax.set_xticklabels(range(env.cols))
        ax.set_yticklabels(range(env.rows))
        plt.colorbar(im, ax=ax, shrink=0.8)

    # ------------------------------------------
    # 2：
    # ------------------------------------------
    ax_path = fig.add_subplot(2, 3, 5)
    # 
    grid = np.zeros((env.rows, env.cols))
    for obs in env.obstacles:
        grid[obs] = -1
    grid[env.goal] = 2

    ax_path.imshow(grid, cmap='Set3', aspect='equal', vmin=-2, vmax=3)

    # 
    path = extract_optimal_path(Q, env)
    path_rows = [p[0] for p in path]
    path_cols = [p[1] for p in path]
    ax_path.plot(path_cols, path_rows, 'b-o', linewidth=2.5, markersize=10)

    # 、、
    ax_path.text(0, 0, 'S', ha='center', va='center', fontsize=16,
                 fontweight='bold', color='green')
    ax_path.text(3, 3, 'G', ha='center', va='center', fontsize=16,
                 fontweight='bold', color='red')
    for obs in env.obstacles:
        ax_path.text(obs[1], obs[0], 'X', ha='center', va='center',
                     fontsize=16, fontweight='bold', color='black')

    # 
    for idx, (r, c) in enumerate(path):
        ax_path.text(c, r, str(idx), ha='center', va='center',
                     fontsize=8, color='white',
                     bbox=dict(boxstyle='round,pad=0.2', fc='blue', alpha=0.5))

    ax_path.set_title('', fontsize=12)
    ax_path.set_xticks(range(env.cols))
    ax_path.set_yticks(range(env.rows))
    ax_path.set_xticklabels(range(env.cols))
    ax_path.set_yticklabels(range(env.rows))
    ax_path.grid(True, alpha=0.3)

    # ------------------------------------------
    # 3：
    # ------------------------------------------
    ax_reward = fig.add_subplot(2, 3, 6)
    ax_reward.plot(episode_rewards, alpha=0.3, color='lightblue', label='')
    # 
    window = 20
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards,
                                 np.ones(window) / window, mode='valid')
        ax_reward.plot(range(window - 1, len(episode_rewards)),
                       moving_avg, color='blue', linewidth=2,
                       label=f'{window}')
    ax_reward.set_xlabel('', fontsize=11)
    ax_reward.set_ylabel('', fontsize=11)
    ax_reward.set_title('', fontsize=12)
    ax_reward.legend(fontsize=9)
    ax_reward.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/gridworld_q_learning_results.png', dpi=150, bbox_inches='tight')
    print("\n output/gridworld_q_learning_results.png")
    plt.show()


# ==========================================
# ：
# ==========================================
def main():
    """： →  →  Q  → """

    #  GridWorld 
    env = GridWorld()
    print("GridWorld ")
    print(f"  : {env.start}")
    print(f"  : {env.goal}")
    print(f"  : {env.obstacles}")
    print(f"  : {env.action_names}")

    #  Q-Learning
    Q, episode_rewards, episode_steps = train_q_learning(env, n_episodes=500)

    #  Q 
    print_q_table(Q, env)

    # 
    path = extract_optimal_path(Q, env)
    print(f"\n: {' → '.join([str(p) for p in path])}")
    print(f": {len(path) - 1} ")

    # 
    total_r = 0
    for i in range(len(path) - 1):
        if i < len(path) - 2:
            total_r += -1  # ：-1
        else:
            total_r += 10  # ：+10
    print(f": {total_r}")

    # 
    visualize_results(Q, episode_rewards, env)


if __name__ == "__main__":
    main()
