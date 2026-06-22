"""
5：Actor-Critic  —— CartPole-v1
， REINFORCE 

REINFORCE ：
    - （Monte Carlo ）
    - ，
    -  G_t 

Actor-Critic ：
    -  TD(0) 
    - advantage = r + γ * V(s') - V(s)
    - ，
    -  V(s') （bootstrap），

：
     Actor-Critic 
    - ：（，）
    - Actor ：（）
    - Critic ：（）

：
    python actor_critic_cartpole.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# 
os.makedirs("output", exist_ok=True)

# 
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# ：Actor-Critic 
# ==========================================
class ActorCritic(nn.Module):
    """
    Actor-Critic ：，

    ：
         state (=4)
            │
        ┌───────┐
        │ Linear│ 4 → 128
        │  ReLU │
        └───────┘
            │
        ┌───────┐
        │  Actor │ 128 → 2 → Softmax  （：）
        └───────┘
            │
        ┌───────┐
        │ Critic │ 128 → 1            （：）
        └───────┘

    ：
        - 
        - Actor  Critic 
        - ，
    """

    def __init__(self, state_dim=4, action_dim=2, hidden_dim=128):
        super(ActorCritic, self).__init__()

        # ：
        self.shared_backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor ：
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
        )

        # Critic ：（）
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        """
        ，

        ：
            x:  [batch_size, state_dim]
        ：
            probs:  [batch_size, action_dim]
            value:  [batch_size]
        """
        # 
        features = self.shared_backbone(x)

        # Actor 
        action_logits = self.actor_head(features)
        probs = torch.softmax(action_logits, dim=-1)

        # Critic 
        value = self.critic_head(features).squeeze(-1)

        return probs, value


# ==========================================
# ： TD 
# ==========================================
def compute_advantage(reward, value, next_value, gamma=0.99, done=False):
    """
     TD(0) 

    TD  = r + γ * V(s') - V(s)

    ：
        - V(s)  Critic ""
        - r + γ * V(s') " + "
        - ""：（）（）

     REINFORCE ：
        REINFORCE: advantage = G_t（）
        Actor-Critic: advantage = r + γ * V(s') - V(s)（ TD ）

    ：
        reward:  r_t
        value:  V(s_t)
        next_value:  V(s_{t+1})
        gamma: 
        done: 
    ：
        advantage: TD 
    """
    if done:
        # ，， = r_t
        target = reward
    else:
        # TD ：r_t + γ * V(s_{t+1})
        target = reward + gamma * next_value

    advantage = target - value
    return advantage, target


# ==========================================
# ：
# ==========================================
def train():
    """
    Actor-Critic 

    （ REINFORCE ）：
        REINFORCE： →  G_t → 
        Actor-Critic： TD  → 

     Actor-Critic （online learning），
    ，。
    """
    # ----------  ----------
    num_episodes = 500
    gamma = 0.99
    learning_rate = 1e-3
    hidden_dim = 128

    # ----------  ----------
    env = gym.make("CartPole-v1")
    model = ActorCritic(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dim=hidden_dim,
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 
    episode_rewards = []
    episode_actor_losses = []
    episode_critic_losses = []

    print("=" * 60)
    print("  Actor-Critic —— CartPole-v1 ")
    print("=" * 60)
    print(f"  :")
    print(f"    : {num_episodes}")
    print(f"     γ: {gamma}")
    print(f"    : {learning_rate}")
    print(f"    : {hidden_dim}")
    print("=" * 60)

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        total_actor_loss = 0
        total_critic_loss = 0
        steps = 0

        done = False
        truncated = False

        while not (done or truncated):
            # ========== ： ==========
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # ：
            probs, value = model(state_tensor)
            probs = probs.squeeze(0)     # [action_dim]
            value = value.squeeze()       # 

            # ========== ：（） ==========
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            #  log π(a|s)，
            log_prob = dist.log_prob(action)

            # ========== ：， ==========
            next_state, reward, done, truncated, _ = env.step(action.item())
            episode_reward += reward
            steps += 1

            # ========== ： ==========
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            with torch.no_grad():
                _, next_value = model(next_state_tensor)
                next_value = next_value.squeeze()

            # ========== ： TD  ==========
            # TD ：A(s,a) = r + γ * V(s') - V(s)
            is_done = done or truncated
            advantage, target = compute_advantage(
                reward, value, next_value, gamma, done=is_done
            )

            # Actor ：-log π(a|s) * A(s,a)
            #  REINFORCE ， advantage  TD 
            actor_loss = -log_prob * advantage

            # Critic ： V(s)  TD  r + γ * V(s')
            critic_loss = nn.MSELoss()(value, target.detach())

            # （，）
            total_loss = actor_loss + critic_loss

            # ========== ： ==========
            # ：REINFORCE ，！
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_losses_save = critic_loss.item()
            total_critic_loss += total_critic_losses_save

            # 
            state = next_state

        # 
        episode_rewards.append(episode_reward)
        episode_actor_losses.append(total_actor_loss / max(steps, 1))
        episode_critic_losses.append(total_critic_loss / max(steps, 1))

        #  50 
        if (episode + 1) % 50 == 0:
            recent_avg = np.mean(episode_rewards[-50:])
            print(
                f"   {episode + 1:4d}/{num_episodes} | "
                f": {episode_reward:6.1f} | "
                f"50: {recent_avg:6.1f} | "
                f": {steps:3d}"
            )

    env.close()

    # ----------  ----------
    print("=" * 60)
    print("  ！")
    print(f"   50 : {np.mean(episode_rewards[-50:]):.1f}")
    print(f"  : {np.max(episode_rewards):.1f}")
    print("=" * 60)

    # ----------  REINFORCE  ----------
    compare_with_reinforce(episode_rewards)

    # ----------  ----------
    plot_training_curve(episode_rewards)


# ==========================================
# ： REINFORCE 
# ==========================================
def compare_with_reinforce(actor_critic_rewards):
    """
     Actor-Critic  REINFORCE 

    ：（ 195）
    CartPole-v1 ""： 100  >= 195
    """
    target_reward = 195

    #  Actor-Critic 
    ac_solve_episode = None
    for i in range(len(actor_critic_rewards) - 99):
        window_avg = np.mean(actor_critic_rewards[i:i + 100])
        if window_avg >= target_reward:
            ac_solve_episode = i + 100
            break

    print("\n" + "-" * 60)
    print("  ")
    print("-" * 60)
    print(f"  CartPole-v1 : 100 >= {target_reward}")

    if ac_solve_episode:
        print(f"  Actor-Critic  {ac_solve_episode} ")
    else:
        print(f"  Actor-Critic  {len(actor_critic_rewards)} ")

    #  REINFORCE 
    print(f"\n  【】:")
    print(f"    REINFORCE  300-500+  CartPole")
    print(f"    Actor-Critic  200-350 ")
    print(f"    ：Actor-Critic ，")
    print(f"           TD(0)  Monte Carlo ")
    print("-" * 60)


# ==========================================
# ：
# ==========================================
def plot_training_curve(episode_rewards):
    """
     Actor-Critic 

    
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # 
    ax.plot(episode_rewards, alpha=0.3, color='steelblue', label='（）')

    # 
    window = 50
    moving_avg = [np.mean(episode_rewards[max(0, i - window + 1):i + 1])
                  for i in range(len(episode_rewards))]
    ax.plot(moving_avg, color='crimson', linewidth=2.0,
            label=f'（={window}）')

    # 
    ax.axhline(y=195, color='green', linestyle='--', alpha=0.7,
               label='（=195）')

    ax.set_xlabel('', fontsize=12)
    ax.set_ylabel('', fontsize=12)
    ax.set_title('Actor-Critic —— CartPole-v1 ', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/actor_critic_cartpole_rewards.png', dpi=150, bbox_inches='tight')
    print("   output/actor_critic_cartpole_rewards.png")
    plt.show()


# ==========================================
# 
# ==========================================
if __name__ == "__main__":
    train()
