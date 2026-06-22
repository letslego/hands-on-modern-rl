# 5.1 ：

，——，。 30%， 70%。， 1 ，。

？——70%  30%。 AI ？，。

::: tip  3 
 3 [](../chapter03_mdp/bandit) Regret——""。****：，Regret ？****： AI 。——[](../chapter03_mdp/bandit) $\mathbb{E}[R_a]$。
:::

——。，""（），。，""""。，————：**""**。

 3 。[](../chapter03_mdp/mdp)（""）， AI [](../chapter03_mdp/policy-objective) $\pi_\theta(a|s)$。

## 

```
┌──────────────────────────────────┐
│                                  │
│   ┌───┐          ┌───┐           │
│   │ A │          │ B │           │
│   │🔴│          │🔵│           │
│   └─┬─┘          └─┬─┘           │
│     │  30%      │  70%    │
│     │               │            │
│     └───────────────┘            │
│                                  │
│   ：， +1     │
│   ： AI  B      │
└──────────────────────────────────┘
```

##  PyTorch 

—— Softmax 。（），：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# ==========================================
# 1. ： Softmax 
# ==========================================
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 2)  # 1（），2（AB）

    def forward(self, x):
        logits = self.linear(x)
        return torch.softmax(logits, dim=-1)  # Softmax 

policy = PolicyNetwork()
optimizer = optim.Adam(policy.parameters(), lr=0.01)

# ==========================================
# 2. ：
# ==========================================
win_probs = [0.3, 0.7]  # A: 30%, B: 70%

def pull_arm(action):
    return 1.0 if random.random() < win_probs[action] else 0.0

# ==========================================
# 3. REINFORCE （，）
# ==========================================
prob_history = []
num_episodes = 300

for ep in range(num_episodes):
    state = torch.tensor([1.0])

    # ，
    probs = policy(state)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()          # 
    log_prob = dist.log_prob(action)  # log π(a|s)

    # ，
    reward = pull_arm(action.item())

    # REINFORCE ： → 
    loss = -log_prob * reward

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 
    with torch.no_grad():
        prob_history.append(policy(state)[1].item())  #  B 

print(f" P(B): {prob_history[0]:.3f}")
print(f" P(B): {prob_history[-1]:.3f}")
```

 `loss = -log_prob * reward` 。：（reward=1），`-log_prob * 1` ，。（reward=0），，。 PyTorch （），（）。

—— 3 [](../chapter03_mdp/policy-objective) $\nabla_\theta J(\theta) \propto \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot G_t]$ 。。

## 

，：

```
 B 

 1.0 ┤
     │                    ╱━━━━━━━━━━━━━━━━━━  ← ： 0.85-0.95
 0.9 ┤                ╱━╱
     │            ╱╱╱╱╱
 0.8 ┤        ╱╱╱╱
     │     ╱╱╱╱              ← ： B 
 0.7 ┤  ╱╱╱╱
     │╱╱╱╱
 0.6 ┤╲╱
     │ ╲ ╱╲  ╱
 0.5 ┤─╲╱╲╱╲╱╲────────────  ← ： 0.5 （）
     └────────────────────────────────────────
     0    50   100  150  200  250  300
                  Episode
```

：（""，）； B （" B "）；（" B "）。

：，。——****。

## 

。，：

-  B （30% ）， "reward = 0" ， B —— B 
-  A （30% ）， "reward = 1" ， A —— A 

""，。——，。

 0.01  0.1， A  B —— A  A， B  B，。——，。

## 

 RL ——：

- ****：（）——""
- ****：（）——" B "
- ****：，

。，（）；，（）。，。

>  7  [PPO](../chapter07_ppo/intro)  entropy bonus（）——，""。

<details>
<summary>： B  55%（ 70%），？</summary>

，、。 A  B （55% vs 30%），""。——""""，。（Baseline）。

</details>

。 `-log_prob * reward` ？？——[ REINFORCE](./policy-gradient)——。
