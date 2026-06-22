# C.8 DAPO

DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization） 2025  GRPO ， RL ：、、Token  loss、。。

---

## 

GRPO ：；/ prompt ； loss ； 0 ，。DAPO 。

## 

- `ratio`： $r = \exp(\text{new\_logp} - \text{old\_logp})$
- `advantage`： z-score $\hat{A}_{i,t} = (R_i - \bar R)/\mathrm{std}(R)$（ GRPO ）
- `ε_{low}`、`ε_{high}`：/（ 0.2 / 0.28）
- `reward_std`： reward ，
- `max_len`、`buffer_len`、`penalty_factor`：、、

## 

> **：clip  ε、/ prompt 、loss  token 、。**

---

## 

|       | GRPO                                | DAPO                                              |
| --------- | ----------------------------------- | ------------------------------------------------- |
|       |  `clip(r, 1-ε, 1+ε)`            |  `clip(r, 1-ε_{low}, 1+ε_{high})`，   |
|       |  prompt                     |  reward  0（/） prompt    |
| loss  | （ token-mean  seq-mean） | token （ token  token ）  |
|   |  reward=0（）           | ：， `penalty_factor` |

---

## （Clip-Higher）

### 

 `clip(r, 1-ε, 1+ε)`  ε。 advantage ， ε ； advantage ， ε 。DAPO ，。

### 

> **PPO  min-clipped， $\varepsilon$： $\varepsilon_{high}$ 、 $\varepsilon_{low}$ 。**

### 

```
ratio = exp(new_logp - old_logp)

# PPO  min-clipped， clip  ε
surr1   = ratio * advantage
surr2   = clip(ratio, 1 - eps_low, 1 + eps_high) * advantage
loss_t  = -min(surr1, surr2)            # per-token loss
```

> ： advantage > 0 （ min ），advantage < 0 ，"、"， PPO  min-clipped 。

### Python 

```python
import numpy as np

def dapo_policy_loss(new_logp, old_logp, advantages,
                     eps_low=0.2, eps_high=0.28):
    """
    new_logp:   [T]
    old_logp:   [T]
    advantages: [T]
     token  loss（ token ）
    """
    ratio = np.exp(new_logp - old_logp)
    surr1 = ratio * advantages
    surr2 = np.clip(ratio, 1 - eps_low, 1 + eps_high) * advantages
    loss_per_token = -np.minimum(surr1, surr2)
    return loss_per_token.sum() / len(loss_per_token)
```

### PyTorch 

```python
import torch

def dapo_policy_loss(new_logps, old_logps, advantages,
                     eps_low=0.2, eps_high=0.28):
    """
    new_logps:  [B, seq_len]
    old_logps:  [B, seq_len]
    advantages: [B, seq_len]
    """
    ratio = torch.exp(new_logps - old_logps)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * advantages
    loss_per_token = -torch.minimum(surr1, surr2)
    return loss_per_token.sum() / loss_per_token.numel()
```

---

## （Dynamic Sampling）

### 

GRPO  advantage  z-score。 prompt  G ， reward  0，z-score ， prompt 。DAPO  prompt， batch 。

### 

> ** reward  0（/）→  → ， batch。**

### 

```
#  prompt  G 
rewards = [get_reward(c) for c in group]      # [G]

#  reward  → ， prompt 
if std(rewards) == 0:
    skip this prompt and resample
```

### PyTorch 

```python
def dynamic_sampling_filter(rewards):
    """
    rewards: [B, G]  B  prompt， G  reward
     bool mask [B]，True = 
    """
    return rewards.std(dim=1) > 1e-6
```

---

## Token  Loss

### 

GRPO  loss： token  loss，。（token ） loss ， token 。DAPO  token ： token ， token ，。

### 

> ** seq-mean—— token ，。**

### 

```
loss_mat = -min(ratio*A, clip(ratio, 1-eps_low, 1+eps_high)*A)   # [B, T]

# GRPO:  token-mean  seq-mean  →  
seq_loss = mean(loss_mat, dim=token)        # 
loss_grpo = mean(seq_loss)

# DAPO:  token 
loss_dapo = sum(loss_mat) / total_num_tokens
```

### PyTorch 

```python
def token_level_loss(loss_mat, loss_mask):
    """
    loss_mat:  [B, T]  per-token policy loss
    loss_mask: [B, T]  1 for valid token, 0 for padding
     token  loss
    """
    return (loss_mat * loss_mask).sum() / loss_mask.sum()
```

---

## （Soft Overlong Punishment）

### 

GRPO ，——""，""。DAPO ： `[max_len - buffer_len, max_len]` ， `max_len - buffer_len` ， `-penalty_factor` ，。

### 

> ** `max_len − buffer_len` ，`-penalty_factor` ——。**

### 

```
expected_len = max_len - buffer_len
exceed_len   = response_length - expected_len

if exceed_len > 0:
    # ：， -penalty_factor（）
    penalty = max(-penalty_factor, -(exceed_len / buffer_len) * penalty_factor)
    reward = reward + penalty        # penalty ≤ 0
```

### Python 

```python
def soft_overlong_penalty(response_length, max_len,
                          buffer_len, penalty_factor=1.0):
    """（≤0）， reward """
    expected_len = max_len - buffer_len
    exceed_len = response_length - expected_len
    if exceed_len <= 0:
        return 0.0
    linear = -(exceed_len / buffer_len) * penalty_factor
    return max(-penalty_factor, linear)        # ，
```

---

## DAPO  Loss 

```
# 1.  z-score （ GRPO ）
advantages = (rewards - rewards.mean(dim=G)) / (rewards.std(dim=G) + eps)

# 2. 
valid = dynamic_sampling_filter(rewards)        # / prompt

# 3.  + token  loss
ratio = exp(new_logp - old_logp)
surr1 = ratio * advantages
surr2 = clip(ratio, 1 - eps_low, 1 + eps_high) * advantages
loss_mat = -minimum(surr1, surr2)               # per-token

# 4. token （，）
policy_loss = (loss_mat * mask)[valid].sum() / mask[valid].sum()

# 5. KL （ GRPO ）
kl = ((exp(ref_logp - new_logp) - 1) - (ref_logp - new_logp)).mean()

loss = policy_loss + kl_coeff * kl
```

---

## 

|                                 |                                                                    |
| ----------------------------------- | ---------------------------------------------------------------------- |
|  ≠                  |  PPO  `min(r*A, clip(r,lo,hi)*A)`， ε          |
| / advantage  ε  | A>0 、A<0 （min ），"、" |
|                       | "reward "，" reward ** 0**"                  |
| Token  loss       | GRPO ，DAPO  token ，                |
|             |  `exceed_len / buffer_len`， `-penalty_factor`               |
| Advantage             |  GRPO ，DAPO                                       |
