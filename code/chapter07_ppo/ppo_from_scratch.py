"""
6： PPO（）
—— PyTorch  CartPole-v1  PPO 

PPO ：
    ratio = exp(new_logprob - old_logprob)
    clipped_ratio = clip(ratio, 1-eps, 1+eps)
    policy_loss = -min(ratio * advantage, clipped_ratio * advantage)

：
    python ppo_from_scratch.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Categorical

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
    Actor-Critic ： + 

    ：
        :  state_dim → 64 → 64 (ReLU)
        Actor:   64 → action_dim ( logits)
        Critic:  64 → 1 ( V(s))

    ：
        - ，
        - Actor  Critic 
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()

        # 
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Actor ： logits
        self.actor_head = nn.Linear(64, action_dim)

        # Critic ：
        self.critic_head = nn.Linear(64, 1)

    def forward(self, x):
        """，"""
        shared_features = self.shared_net(x)

        # Actor: 
        action_logits = self.actor_head(shared_features)
        action_probs = F.softmax(action_logits, dim=-1)

        # Critic: 
        value = self.critic_head(shared_features)

        return action_probs, value

    def get_action(self, state):
        """，、log、"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs, value = self.forward(state_tensor)

        #  Categorical 
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob, value.squeeze()

    def evaluate(self, states, actions):
        """
         (, ) 
        ：log、、
        """
        action_probs, values = self.forward(states)
        dist = Categorical(action_probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values.squeeze(), entropy


# ==========================================
# ：GAE（）
# ==========================================
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
     (Generalized Advantage Estimation)

    GAE ：
        δ_t = r_t + γ * V(s_{t+1}) - V(s_t)    # TD 
        A_t = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}      # GAE 

    ：
        rewards: 
        values:   V(s)
        dones:   
        gamma:   （）
        lam:     GAE lambda（-）
            λ=0: 、（ TD ）
            λ=1: 、（）

    ：
        advantages: 
        returns:    （ Critic）
    """
    advantages = []
    gae = 0

    # 
    values = list(values)
    #  V(s)=0
    next_value = 0

    #  GAE
    for t in reversed(range(len(rewards))):
        if dones[t]:
            # ， 0
            next_value = 0
            gae = 0

        # TD ：δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * next_value - values[t]

        # GAE ：A_t = δ_t + (γλ) * A_{t+1}
        gae = delta + gamma * lam * gae

        advantages.insert(0, gae)

        #  V(s)
        next_value = values[t]

    advantages = torch.FloatTensor(advantages)
    returns = advantages + torch.FloatTensor(values)

    # （）
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


# ==========================================
# ：PPO 
# ==========================================
def ppo_clip_loss(old_logprobs, new_logprobs, advantages, clip_eps=0.2):
    """
    PPO 

    ：
        ratio = exp(new_logprob - old_logprob) = π_new(a|s) / π_old(a|s)
        L_CLIP = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)

     ratio > 1+ε  ratio < 1-ε ，
    → 

    ：
        old_logprobs:  log 
        new_logprobs:  log 
        advantages:   
        clip_eps:      ε（ 0.2）

    ：
        policy_loss: 
        clip_frac:   （）
    """
    # 
    ratio = torch.exp(new_logprobs - old_logprobs)

    # 
    surr1 = ratio * advantages

    # 
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages

    # （）
    policy_loss = -torch.min(surr1, surr2).mean()

    # （）
    with torch.no_grad():
        clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()

    return policy_loss, clip_frac


# ==========================================
# ：
# ==========================================
def collect_trajectories(model, env, n_steps=2048):
    """
     n_steps 

    ：
        - states:  
        - actions: 
        - logprobs:  log （ PPO ）
        - rewards: 
        - dones:   
        - values:  

    ：
        batch  + 
    """
    states = []
    actions = []
    old_logprobs = []
    rewards = []
    dones = []
    values = []

    obs, _ = env.reset()
    episode_rewards = []
    current_ep_reward = 0

    for step in range(n_steps):
        state_tensor = torch.FloatTensor(obs)

        # 
        with torch.no_grad():
            action_probs, value = model(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        # 
        states.append(obs.copy())
        actions.append(action.item())
        old_logprobs.append(log_prob.item())
        values.append(value.item())

        # 
        next_obs, reward, done, truncated, _ = env.step(action.item())
        rewards.append(reward)
        dones.append(done or truncated)

        current_ep_reward += reward

        if done or truncated:
            episode_rewards.append(current_ep_reward)
            current_ep_reward = 0
            next_obs, _ = env.reset()

        obs = next_obs

    # 
    batch = {
        "states": torch.FloatTensor(np.array(states)),
        "actions": torch.LongTensor(actions),
        "old_logprobs": torch.FloatTensor(old_logprobs),
        "rewards": rewards,
        "dones": dones,
        "values": values,
    }

    return batch, episode_rewards


# ==========================================
# ：PPO 
# ==========================================
def ppo_update(model, optimizer, batch, n_epochs=10, batch_size=64,
               clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
    """
     PPO 

    ：
        1.  →  log_probs
        2.  PPO （）
        3. （Critic）
        4. （）
        5.  =  +  - 

    ：
        （）
    """
    #  GAE 
    advantages, returns = compute_gae(
        batch["rewards"], batch["values"], batch["dones"],
        gamma=0.99, lam=0.95
    )

    #  CPU（）
    states = batch["states"]
    actions = batch["actions"]
    old_logprobs = batch["old_logprobs"]

    dataset_size = states.shape[0]
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    total_clip_frac = 0
    update_count = 0

    for epoch in range(n_epochs):
        # 
        indices = torch.randperm(dataset_size)

        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            mb_indices = indices[start:end]

            mb_states = states[mb_indices]
            mb_actions = actions[mb_indices]
            mb_old_logprobs = old_logprobs[mb_indices]
            mb_advantages = advantages[mb_indices]
            mb_returns = returns[mb_indices]

            # 
            new_logprobs, new_values, entropy = model.evaluate(mb_states, mb_actions)

            # ---- （PPO-Clip）----
            policy_loss, clip_frac = ppo_clip_loss(
                mb_old_logprobs, new_logprobs, mb_advantages, clip_eps
            )

            # ----  ----
            value_loss = F.mse_loss(new_values, mb_returns)

            # ----  ----
            entropy_bonus = entropy.mean()

            # ----  ----
            #  =  + vf_coef *  - ent_coef * 
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_bonus

            # 
            optimizer.zero_grad()
            loss.backward()
            # （）
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            # 
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy_bonus.item()
            total_clip_frac += clip_frac
            update_count += 1

    # 
    metrics = {
        "policy_loss": total_policy_loss / update_count,
        "value_loss": total_value_loss / update_count,
        "entropy": total_entropy / update_count,
        "clip_fraction": total_clip_frac / update_count,
    }

    return metrics


# ==========================================
# ：
# ==========================================
def train():
    """PPO """
    print("=" * 50)
    print("6： PPO — CartPole-v1")
    print("=" * 50)

    # 
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]   # 4
    action_dim = env.action_space.n               # 2

    # 
    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    print(f"\n:")
    print(model)
    print(f"\n: {state_dim}, : {action_dim}")

    # 
    n_steps = 2048        # 
    n_epochs = 10         # 
    batch_size = 64       # 
    clip_eps = 0.2        # PPO 
    total_episodes = 1000 # 

    # 
    all_rewards = []
    all_policy_losses = []
    all_value_losses = []
    all_entropies = []
    all_clip_fracs = []

    print(f"\n（: {total_episodes} ）...")
    print("-" * 50)

    episode_count = 0
    iteration = 0

    while episode_count < total_episodes:
        iteration += 1

        # ：
        batch, ep_rewards = collect_trajectories(model, env, n_steps=n_steps)
        episode_count += len(ep_rewards)
        all_rewards.extend(ep_rewards)

        # ：PPO 
        metrics = ppo_update(
            model, optimizer, batch,
            n_epochs=n_epochs,
            batch_size=batch_size,
            clip_eps=clip_eps,
        )

        all_policy_losses.append(metrics["policy_loss"])
        all_value_losses.append(metrics["value_loss"])
        all_entropies.append(metrics["entropy"])
        all_clip_fracs.append(metrics["clip_fraction"])

        # 
        if iteration % 5 == 0 or len(ep_rewards) > 0:
            recent_rewards = all_rewards[-20:] if len(all_rewards) >= 20 else all_rewards
            avg_reward = np.mean(recent_rewards)
            print(
                f"   {iteration:3d} | "
                f": {episode_count:4d} | "
                f": {avg_reward:6.1f} | "
                f": {metrics['policy_loss']:.4f} | "
                f": {metrics['value_loss']:.4f} | "
                f": {metrics['entropy']:.3f} | "
                f": {metrics['clip_fraction']:.3f}"
            )

    print("-" * 50)
    print(f"！ {episode_count} ，{iteration} ")

    # 
    test_rewards = []
    for _ in range(20):
        obs, _ = env.reset()
        done, truncated = False, False
        total_reward = 0
        while not (done or truncated):
            state_tensor = torch.FloatTensor(obs)
            with torch.no_grad():
                action_probs, _ = model(state_tensor)
            action = torch.argmax(action_probs).item()
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
        test_rewards.append(total_reward)

    mean_reward = np.mean(test_rewards)
    std_reward = np.std(test_rewards)
    print(f"\n20 :  = {mean_reward:.1f} ± {std_reward:.1f}")

    env.close()

    # ==========================================
    # ：
    # ==========================================
    print("\n...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PPO  — CartPole-v1 ", fontsize=16, fontweight="bold")

    # 1：
    ax1 = axes[0, 0]
    window = min(20, len(all_rewards))
    if window > 0:
        smoothed = np.convolve(all_rewards, np.ones(window) / window, mode="valid")
        ax1.plot(range(len(all_rewards)), all_rewards, alpha=0.3, color="#90CAF9", label="")
        ax1.plot(range(window - 1, len(all_rewards)), smoothed, color="#2196F3",
                 linewidth=2, label=f" (={window})")
        ax1.axhline(y=475, color="green", linestyle="--", alpha=0.5, label=" (475)")
    ax1.set_title("", fontsize=13)
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2： & 
    ax2 = axes[0, 1]
    if all_policy_losses:
        ax2.plot(all_policy_losses, color="#F44336", alpha=0.8, linewidth=1.2, label="")
        ax2.plot(all_value_losses, color="#2196F3", alpha=0.8, linewidth=1.2, label="")
    ax2.set_title("", fontsize=13)
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3：
    ax3 = axes[1, 0]
    if all_entropies:
        ax3.plot(all_entropies, color="#FF9800", alpha=0.8, linewidth=1.5)
        ax3.set_title("（）", fontsize=13)
        ax3.set_xlabel("")
        ax3.set_ylabel("")
        ax3.annotate(" = ", xy=(len(all_entropies) * 0.6, max(all_entropies) * 0.8),
                     fontsize=10, color="gray", style="italic")
    ax3.grid(True, alpha=0.3)

    # 4：
    ax4 = axes[1, 1]
    if all_clip_fracs:
        ax4.plot(all_clip_fracs, color="#9C27B0", alpha=0.8, linewidth=1.5)
        ax4.axhline(y=0.2, color="gray", linestyle="--", alpha=0.5, label="clip_range = 0.2")
        ax4.set_title("", fontsize=13)
        ax4.set_xlabel("")
        ax4.set_ylabel("")
        ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/ppo_from_scratch_curves.png", dpi=150, bbox_inches="tight")
    print(": output/ppo_from_scratch_curves.png")
    plt.show()


# ==========================================
# 
# ==========================================
if __name__ == "__main__":
    train()
