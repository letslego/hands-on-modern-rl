# C.4 GRPO  Reward Model

---

## GRPO Loss

****：PPO  policy  Critic  $V(s)$  baseline。GRPO " prompt  G "， reward  advantage， Critic。

****：

- `rewards`： prompt  G （ RM ）， `[G]`
- `advantages`：，$A_i = (r_i - \bar r)/\mathrm{std}(r)$
- `old_log_probs` / `new_log_probs`： log 
- `ref_log_probs`： log ， KL 
- `clip_eps`、`kl_coeff`：PPO 

### 

> ** prompt  G ， reward  z-score  advantage； PPO  clip  KL， Critic。**

### 

```
#  1 ： prompt  G ，
rewards = [reward_fn(generate(prompt)) for _ in range(G)]   # [G]

#  2 ：（） advantage
advantages = (rewards - mean(rewards)) / (std(rewards) + eps)

#  3 ：PPO clipped loss（advantage  2 ， Critic）
ratio = exp(new_logp - old_logp)
surr1 = ratio * advantages
surr2 = clip(ratio, 1-eps, 1+eps) * advantages
policy_loss = -min(surr1, surr2).mean()

#  4 ：k3 KL （，）
log_ratio = ref_logp - new_logp
kl = (exp(log_ratio) - 1 - log_ratio).mean()

#  5 ： loss
loss = policy_loss + kl_coeff * kl
```

### PPO vs GRPO 

|                | PPO                         | GRPO                           |
| -------------- | --------------------------- | ------------------------------ |
| Advantage  | Critic  $V(s)$ → GAE    |  reward              |
|        | 4（actor, critic, ref, rm） | 2~3（actor, ref, rm/verifier） |
| KL             |                         |                        |
|        |  rollout                |  prompt  G               |

### Python 

```python
import numpy as np

def grpo_advantages(rewards):
    """
    rewards: [num_prompts, G]   prompt  G  reward
     z-score  advantage
    """
    mean = rewards.mean(axis=1, keepdims=True)
    std = rewards.std(axis=1, keepdims=True)
    return (rewards - mean) / (std + 1e-8)

def grpo_policy_loss(new_logps, old_logps, advantages, clip_eps=0.2):
    """ PPO clipped loss """
    ratio = np.exp(new_logps - old_logps)
    surr1 = ratio * advantages
    surr2 = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -np.minimum(surr1, surr2).mean()
```

### PyTorch 

```python
import torch
import torch.nn.functional as F

def grpo_loss(log_probs, old_log_probs, ref_log_probs,
              rewards, clip_eps=0.2, kl_coeff=0.05):
    """
    log_probs:     [B, G]   completion  log_prob
    old_log_probs: [B, G]  
    ref_log_probs: [B, G]  
    rewards:       [B, G]   reward
    B = num_prompts, G = group_size
    """
    # 1. （ prompt ）
    advantages = (rewards - rewards.mean(dim=1, keepdim=True)) \
                 / (rewards.std(dim=1, keepdim=True) + 1e-8)   # [B, G]

    # 2. Clipped policy loss（ PPO ）
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # 3. KL （k3 ：log_ratio = log(π_ref/π_θ)， π_θ）
    log_ratio = ref_log_probs - log_probs
    kl = (torch.exp(log_ratio) - 1 - log_ratio).mean()

    return policy_loss + kl_coeff * kl
```

---

## Reward Model（Bradley-Terry ）

****：RLHF-PPO  $r \in \mathbb{R}$ ，（"A  B "）。RM ，""。

****：

- `r_chosen` / `r_rejected`：RM /
- Bradley-Terry ：$P(y_w \succ y_l) = \sigma(r_w - r_l)$， sigmoid

### 

> **：`-log_sigmoid(r_chosen - r_rejected)`，。**

### 

```
#  1 ：RM 
r_w = reward_model(chosen_input)     # 
r_l = reward_model(rejected_input)   # 

#  2 ： r_w > r_l， sigmoid 
loss = -log(sigmoid(r_w - r_l))
```

### Python 

```python
import numpy as np

def log_sigmoid(x):
    return -np.logaddexp(0, -x)   #  log σ(x)

def reward_model_loss(r_chosen, r_rejected):
    """r_chosen, r_rejected: [B]"""
    return -log_sigmoid(r_chosen - r_rejected).mean()
```

### PyTorch 

```python
import torch.nn.functional as F

def reward_model_loss(r_chosen, r_rejected):
    """
    r_chosen:   [B]  RM  chosen 
    r_rejected: [B]  RM  rejected 
    """
    return -F.logsigmoid(r_chosen - r_rejected).mean()
```

---

## ：GRPO、PPO-RLHF  RLVR 

|                | PPO-RLHF     | GRPO                      | RLVR                          |
| -------------- | ------------ | ------------------------- | ----------------------------- |
| Advantage  | Critic + GAE |  reward         |  reward             |
| Critic         |          |                     |                         |
| Reward     |  RM  | RM  verifier            | （/） |
|        |          | （ prompt  G ） | （ prompt  G ）     |

RLVR  GRPO ：reward  RM，（、）， reward hacking 。

---

## 

|                            |                                                                          |
| ------------------------------ | ---------------------------------------------------------------------------- |
| GRPO  advantage  | ，** prompt**  G                             |
| GRPO  value loss           |  Critic， value loss， PPO                       |
|      |  std（z-score）； `rewards - mean`（ std）。 |
| RM  policy       |  RM  reward ； policy  RM  detach              |
| KL                     |  completion  token log_prob ， KL， token        |
| k3 KL              | `log_ratio = log(π_ref/π_θ)`， $\pi_\theta$                  |
| RLVR                       | reward （、）， RM                 |
