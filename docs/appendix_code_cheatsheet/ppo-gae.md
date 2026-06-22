# C.2 PPO  GAE

PPO  RL 。 **clipped policy loss**， value loss  GAE。

---

## GAE（）

****： $\hat{A}_t$， actor 。

****：

- `delta_t`：TD  $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$，""
- `advantage_t`： $\hat{A}_t$，
- `gamma`（$\gamma$）：，
- `lambda`（$\lambda$）：-，$\lambda=0$  TD(0)，$\lambda=1$  Monte-Carlo
- `done_t`：episode ， episode 

### 

> **：$\hat{A}_t = \delta_t + \gamma\lambda \hat{A}_{t+1}$。**

### 

```
delta_t     = reward_t + gamma * value_{t+1} * (1 - done_t) - value_t
advantage_t = delta_t + gamma * lambda * (1 - done_t) * advantage_{t+1}
return_t    = advantage_t + value_t
```

$\lambda = 0$  TD（、）；$\lambda = 1$  $\delta$（、）。

### Python 

```python
import numpy as np

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    rewards: [T]
    values:  [T+1]  ( bootstrap value)
    dones:   [T]
    """
    T = len(rewards)
    advantages = np.zeros(T)
    last_adv = 0.0

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        last_adv = delta + gamma * lam * (1 - dones[t]) * last_adv
        advantages[t] = last_adv

    returns = advantages + values[:T]
    return advantages, returns
```

### PyTorch 

```python
import torch

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    rewards: [B, T]
    values:  [B, T+1]
    dones:   [B, T]
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(B)

    for t in reversed(range(T)):
        delta = rewards[:, t] + gamma * values[:, t + 1] * (1 - dones[:, t]) - values[:, t]
        last_adv = delta + gamma * lam * (1 - dones[:, t]) * last_adv
        advantages[:, t] = last_adv

    returns = advantages + values[:, :T]
    return advantages, returns
```

---

## PPO Clipped Policy Loss

****：""， clip  KL ，（）， ratio 。

****：

- `ratio`：$r_t(\theta) = \pi_\theta(a_t\mid s_t) / \pi_{\theta_{old}}(a_t\mid s_t)$，（）
- `advantage`：$\hat{A}_t$， GAE
- `eps`（$\epsilon$）：clip ， `0.1`  `0.2`
- `new_log_prob` / `old_log_prob`： log 

### 

> ** $r_t=\pi_{new}/\pi_{old}$，clip  $[1-\epsilon, 1+\epsilon]$； surrogate  min——。**

$$L^{CLIP} = -\min\big(r_t(\theta) \cdot A_t,\;\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \cdot A_t\big)$$

### 

```
ratio = exp(new_log_prob - old_log_prob)
surr1 = ratio * advantage
surr2 = clip(ratio, 1-eps, 1+eps) * advantage
loss  = -min(surr1, surr2).mean()
```

`advantage > 0`  ratio ，`advantage < 0` ；`min` 。

### Python 

```python
import numpy as np

def ppo_policy_loss(new_logp, old_logp, advantages, clip_eps=0.2):
    """
    new_logp:   [T]   log 
    old_logp:   [T]   log 
    advantages: [T]
    """
    ratio = np.exp(new_logp - old_logp)
    surr1 = ratio * advantages
    surr2 = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -np.minimum(surr1, surr2).mean()
```

### PyTorch 

```python
import torch

def ppo_policy_loss(new_logps, old_logps, advantages, clip_eps=0.2):
    """
    new_logps:   [B, T]
    old_logps:   [B, T]
    advantages:  [B, T]
    """
    ratio = torch.exp(new_logps - old_logps)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -torch.min(surr1, surr2).mean()
```

---

## PPO Value Loss

****： critic  $V(s)$，（value clipping）。

****：

- `value_pred`：critic  $V(s_t)$ 
- `old_values`： critic ， clip 
- `returns`：$R_t = \hat{A}_t + V_{old}(s_t)$，GAE 
- `eps`：clip ， policy loss 

### 

> **$(V_{pred} - R)^2$ ； clip： $\epsilon$。**

### 

```
value_clipped = old_values + clip(value_pred - old_values, -eps, eps)
loss1 = (value_pred    - returns)^2
loss2 = (value_clipped - returns)^2
loss  = 0.5 * max(loss1, loss2).mean()
```

### PyTorch 

```python
def ppo_value_loss(values, old_values, returns, clip_eps=0.2):
    loss1 = (values - returns) ** 2
    values_clipped = old_values + torch.clamp(values - old_values, -clip_eps, clip_eps)
    loss2 = (values_clipped - returns) ** 2
    return 0.5 * torch.max(loss1, loss2).mean()
```

---

## PPO  Loss

```
total_loss = policy_loss + vf_coeff * value_loss - ent_coeff * entropy
```

|                   |         |         |
| --------------------- | ----------- | --------------- |
| policy loss (clipped) |     | `1.0`           |
| value loss (MSE)      |  Critic | `vf_coef=0.5`   |
| entropy bonus         |     | `ent_coef=0.01` |

entropy ： entropy  loss 。

---

## 

|                |                                                                               |
| ------------------ | --------------------------------------------------------------------------------- |
| ratio        |  `exp(logp_new - logp_old)`，                                         |
| ratio      |  $\pi_{old}$，                                |
| clip       |  1  $[1-\epsilon, 1+\epsilon]$， 0  $[-\epsilon, \epsilon]$     |
| min/max        | policy loss ** surrogate**  `min`（）；value loss  MSE  `max` |
|  stop gradient | `old_log_probs`  `old_values`  `.detach()`                                    |
| advantage  |  advantage  batch （mean 0, std 1）                           |
| GAE            | （）                                                |
| GAE  done mask   | `done=1` ：`gamma * lambda * (1-done) * next_adv`                       |
| value  bootstrap | `values`  T+1， bootstrap value                               |
| entropy        | `- ent_coeff * entropy`（entropy ， loss  = ）            |
