"""
1： ——  PyTorch  PPO  CartPole
 SB3  model.learn() 

 SwanLab （、），
 GUI 。

：
    # ： + SwanLab （ GUI，）
    python 2-pytorch_ppo.py

    #  GUI （）
    python 2-pytorch_ppo.py --gui

 --gui ：
     headless（）， GUI 。
    --gui  CartPole 。
     GUI ，（~16ms），；
     GUI ，，。
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import swanlab


# ==========================================
# ：Actor-Critic （ + ）
# ==========================================
class ActorCritic(nn.Module):
    """
     Actor-Critic （ SB3 MlpPolicy ）：
    - Actor  Critic ，
    - ：actor  gain=0.01 
    """

    def __init__(self, obs_dim=4, act_dim=2, hidden=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self._init_weights()

    def _init_weights(self):
        """， SB3 """
        for module in self.actor:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        for module in self.critic:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        # actor  gain → 
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.constant_(self.actor[-1].bias, 0)
        # critic  gain=1
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.constant_(self.critic[-1].bias, 0)

    def forward(self, x):
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value.squeeze(-1)

    def get_action(self, obs, deterministic=False):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value


# ==========================================
# ：（Rollout）
# ==========================================
def collect_rollout(model, env, num_steps=2048):
    """
    ， terminated vs truncated：
    - terminated（）：V(s')=0
    - truncated（）：V(s') bootstrap
    - rollout ： bootstrap
    """
    obs, _ = env.reset()
    transitions = []

    for _ in range(num_steps):
        obs_tensor = torch.FloatTensor(obs)
        with torch.no_grad():
            action, log_prob, value = model.get_action(obs_tensor)

        next_obs, reward, terminated, truncated, _ = env.step(action.item())

        # truncated  terminated →  next_obs  bootstrap
        transitions.append({
            "obs": obs,
            "action": action.item(),
            "log_prob": log_prob.item(),
            "value": value.item(),
            "reward": float(reward),
            "terminated": terminated,
            "truncated": truncated,
            "next_obs": next_obs if truncated and not terminated else None,
        })

        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

    # rollout  bootstrap：， V(s_last)
    if not (terminated or truncated):
        with torch.no_grad():
            _, _, bootstrap_value = model.get_action(torch.FloatTensor(obs))
        last_bootstrap = bootstrap_value.item()
    else:
        last_bootstrap = 0.0

    return transitions, last_bootstrap


# ==========================================
# ： GAE 
# ==========================================
def compute_gae(model, transitions, last_bootstrap, gamma=0.99, lam=0.95):
    """
    ，：
    - terminated（）： GAE，V(s')=0
    - truncated（）： GAE， V(next_obs)  bootstrap
    - ： GAE
    """
    n = len(transitions)
    rewards = [t["reward"] for t in transitions]
    values = [t["value"] for t in transitions]

    #  truncated  bootstrap value
    bootstrap_values = [0.0] * n
    for i, t in enumerate(transitions):
        if t["truncated"] and not t["terminated"] and t["next_obs"] is not None:
            with torch.no_grad():
                _, _, bv = model.get_action(torch.FloatTensor(t["next_obs"]))
            bootstrap_values[i] = bv.item()

    advantages = []
    gae = 0
    next_value = last_bootstrap

    for step in reversed(range(n)):
        t = transitions[step]

        if t["terminated"]:
            # ：V(s') = 0
            delta = rewards[step] - values[step]
            gae = delta
        elif t["truncated"]:
            # ： V(next_obs) bootstrap， GAE
            delta = rewards[step] + gamma * bootstrap_values[step] - values[step]
            gae = delta
        else:
            # 
            delta = rewards[step] + gamma * next_value - values[step]
            gae = delta + gamma * lam * gae

        next_value = values[step]
        advantages.insert(0, gae)

    advantages = torch.FloatTensor(advantages)
    returns = advantages + torch.FloatTensor(values)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


# ==========================================
# ：PPO 
# ==========================================
def ppo_update(model, optimizer, transitions, advantages, returns,
               clip_eps=0.2, epochs=10, batch_size=64):
    """PPO """
    obs = np.array([t["obs"] for t in transitions])
    actions = np.array([t["action"] for t in transitions])
    old_log_probs = np.array([t["log_prob"] for t in transitions])

    obs = torch.FloatTensor(obs)
    actions = torch.LongTensor(actions)
    old_log_probs = torch.FloatTensor(old_log_probs)

    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    total_kl = 0
    total_clip_frac = 0
    n_updates = 0

    for _ in range(epochs):
        indices = np.random.permutation(len(transitions))

        for start in range(0, len(transitions), batch_size):
            idx = indices[start:start + batch_size]

            batch_obs = obs[idx]
            batch_actions = actions[idx]
            batch_old_log_probs = old_log_probs[idx]
            batch_advantages = advantages[idx]
            batch_returns = returns[idx]

            logits, values = model(batch_obs)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(batch_actions)

            # PPO 
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 
            value_loss = ((values - batch_returns) ** 2).mean()

            # （）
            entropy = dist.entropy().mean()

            loss = policy_loss + 0.5 * value_loss - 0.0 * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            # 
            with torch.no_grad():
                total_kl += (batch_old_log_probs - new_log_probs).mean().item()
                total_clip_frac += ((ratio - 1.0).abs() > clip_eps).float().mean().item()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            n_updates += 1

    return {
        "policy_loss": total_policy_loss / n_updates,
        "value_loss": total_value_loss / n_updates,
        "entropy": total_entropy / n_updates,
        "approx_kl": total_kl / n_updates,
        "clip_fraction": total_clip_frac / n_updates,
    }


# ==========================================
# ：
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description=" PyTorch PPO CartPole ")
    parser.add_argument(
        "--gui", action="store_true",
        help=" GUI （，）",
    )
    return parser.parse_args()


def train():
    args = parse_args()
    os.makedirs("output", exist_ok=True)

    env = gym.make("CartPole-v1")

    # （、、）
    print("=" * 50)
    print("CartPole-v1 ")
    print("=" * 50)
    print(f"  :  {env.observation_space}")
    print(f"  :  {env.action_space}")
    print(f"  :  {env.observation_space.high}")
    print(f"  :  {env.observation_space.low}")
    print(f"  :   > ±{env.unwrapped.x_threshold}, "
          f" > ±{env.unwrapped.theta_threshold_radians:.4f} rad "
          f"(≈ ±{np.degrees(env.unwrapped.theta_threshold_radians):.0f}°)")
    print("=" * 50)

    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    total_iterations = 40
    steps_per_rollout = 2048

    #  SwanLab
    swanlab.init(
        project="cartpole-pytorch",
        experiment_name="PPO-PyTorch-CartPole-v1",
        mode="local",
        config={
            "algorithm": "PPO",
            "lr": 3e-4,
            "total_iterations": total_iterations,
            "steps_per_rollout": steps_per_rollout,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "epochs": 10,
            "batch_size": 64,
        },
    )

    print("（ PyTorch PPO + SwanLab）...")
    print("-" * 60)

    total_timesteps = 0

    for iteration in range(total_iterations):
        # 
        transitions, last_bootstrap = collect_rollout(model, env, steps_per_rollout)

        total_timesteps += len(transitions)

        # 
        ep_rewards = []
        ep_lengths = []
        ep_reward = 0
        ep_length = 0
        for t in transitions:
            ep_reward += t["reward"]
            ep_length += 1
            if t["terminated"] or t["truncated"]:
                ep_rewards.append(ep_reward)
                ep_lengths.append(ep_length)
                ep_reward = 0
                ep_length = 0

        # 
        advantages, returns = compute_gae(model, transitions, last_bootstrap)

        # PPO 
        metrics = ppo_update(
            model, optimizer, transitions, advantages, returns
        )

        # （ Critic ， SB3 ）
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(np.array([t["obs"] for t in transitions]))
            _, updated_values = model(obs_tensor)
        return_values = returns.numpy()
        updated_values_np = updated_values.numpy()
        var_returns = np.var(return_values)
        if var_returns < 1e-6:
            # （ 500 ），EV ， 0
            explained_variance = 0.0
        else:
            explained_variance = 1 - np.var(return_values - updated_values_np) / var_returns

        mean_reward = np.mean(ep_rewards) if ep_rewards else 0
        mean_ep_len = np.mean(ep_lengths) if ep_lengths else 0

        # （ SB3 ）
        frac = 1.0 - iteration / total_iterations
        lr = 3e-4 * frac
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        #  SwanLab（ SB3 ）
        swanlab.log({
            "rollout/ep_rew_mean": mean_reward,
            "rollout/ep_len_mean": mean_ep_len,
            "train/policy_gradient_loss": metrics["policy_loss"],
            "train/value_loss": metrics["value_loss"],
            "train/entropy_loss": -metrics["entropy"],
            "train/approx_kl": metrics["approx_kl"],
            "train/clip_fraction": metrics["clip_fraction"],
            "train/clip_range": 0.2,
            "train/explained_variance": explained_variance,
            "train/learning_rate": lr,
            "train/n_updates": (iteration + 1) * 10 * (steps_per_rollout // 64),
            "time/total_timesteps": total_timesteps,
            "time/iterations": iteration + 1,
        }, step=iteration)

        print(
            f"   {iteration + 1:2d}/{total_iterations} | "
            f": {len(ep_rewards):3d} | "
            f": {mean_reward:6.1f} | "
            f"KL: {metrics['approx_kl']:.4f} | "
            f"clip%: {metrics['clip_fraction']:.1%}"
        )

    print("-" * 60)

    # 
    eval_rewards = []
    for _ in range(20):
        obs, _ = env.reset()
        done, truncated, score = False, False, 0
        while not (done or truncated):
            obs_tensor = torch.FloatTensor(obs)
            with torch.no_grad():
                action, _, _ = model.get_action(obs_tensor, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action.item())
            score += reward
        eval_rewards.append(score)

    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    print(f"\n！20 : {mean_reward:.1f} +/- {std_reward:.1f}")

    swanlab.log({
        "eval/mean_reward": mean_reward,
        "eval/std_reward": std_reward,
    })

    # 
    torch.save(model.state_dict(), "output/pytorch_ppo_cartpole.pth")
    print(f" output/pytorch_ppo_cartpole.pth")

    # GUI 
    if args.gui:
        try:
            vis_env = gym.make("CartPole-v1", render_mode="human")
            print("\n（5 ）...")
            for ep in range(5):
                obs, _ = vis_env.reset()
                done, truncated, score = False, False, 0
                while not (done or truncated):
                    obs_tensor = torch.FloatTensor(obs)
                    with torch.no_grad():
                        action, _, _ = model.get_action(obs_tensor, deterministic=True)
                    obs, reward, done, truncated, _ = vis_env.step(action.item())
                    score += reward
                print(f"   {ep + 1} : {score}")
            vis_env.close()
            print("\nGUI 。")
        except Exception:
            print("( GUI ，)")
    else:
        print("\n:  --gui 。")

    env.close()
    swanlab.finish()

    print("SwanLab : swanlab watch swanlog")


if __name__ == "__main__":
    train()
