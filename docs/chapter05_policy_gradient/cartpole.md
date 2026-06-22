# 5.3 ： CartPole

> ****： REINFORCE  `CartPole-v1`，，""。

> ****：[reinforce_cartpole.py](https://github.com/letslego/hands-on-modern-rl/blob/main/code/chapter05_policy_gradient/reinforce_cartpole.py) · [requirements.txt](https://github.com/letslego/hands-on-modern-rl/blob/main/code/chapter05_policy_gradient/requirements.txt)

 REINFORCE 。——、、。：`CartPole-v1`。，。， `+1` ；，episode 。

，CartPole （4 ：、、、），，。（/）， REINFORCE 。

## 

：

```bash
pip install -r code/chapter05_policy_gradient/requirements.txt
```

：

```bash
python code/chapter05_policy_gradient/reinforce_cartpole.py
```

 REINFORCE ，500  episode。：

```python
# ： episode
states, actions, rewards, episode_reward = collect_episode(policy, env)

# ： G_t
returns = compute_returns(rewards, gamma=0.99)

# ：
loss = -(log_probs * returns_tensor).mean()
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

 REINFORCE 。 episode， $G_t$， `loss = -log_prob * G_t` 。

， `output/` 。

## 

![REINFORCE  CartPole-v1 ：](./images/reinforce-cartpole-reward.png)

：

**（episode 0–50）**：，， 10–30 。——""。

**（episode 50–200）**：， episode ，$G_t$ ，。，—— episode 。

**（episode 200–500）**：， 100–200。——。

## 

CartPole ， episode 。，。$G_t$ ——，。

：

- ****：， 200+， 30。，。
- ****：，。 500  episode ， 1000 。
- ****：，；，。。

：REINFORCE  $G_t$ ""， $G_t$ 。，；，。

## 

****——：

```python
def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G  # G_t = r_t + γ * G_{t+1}
        returns.insert(0, G)
    return returns
```

：$G_T = r_T$。：$G_{T-1} = r_{T-1} + \gamma G_T$。。 $G_t$ 。

**， argmax**—— DQN ：

```python
probs = policy(state_tensor)
dist = torch.distributions.Categorical(probs)
action = dist.sample()  # 
```

DQN  `argmax Q`，。REINFORCE ，—— 60% ， 60% 。

**on-policy **——REINFORCE ，：

```python
#  episode 
states, actions, rewards, episode_reward = collect_episode(policy, env)
```

DQN 。REINFORCE  $\mathbb{E}_{\pi_\theta}$  $\pi_\theta$ 。，。 DQN 。

## 

CartPole  REINFORCE ，。 $G_t$ 。： $b(s_t)$， $G_t$  $G_t - b(s_t)$，，。

：[](./pg-improvements)。
