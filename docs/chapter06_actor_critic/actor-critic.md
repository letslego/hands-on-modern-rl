# 6.3 Actor-Critic 

[](./advantage-function) $A(s,a)$  [Critic ](./critic-training)。， Actor  Critic 。

::: tip 

- [ $A(s,a) = Q(s,a) - V(s)$](./advantage-function)——""
- [TD Error $\delta = r + \gamma V(s') - V(s)$](./critic-training)——
- [ $\nabla_\theta J \approx \nabla_\theta \log \pi(a|s) \cdot G_t$](../chapter05_policy_gradient/reinforce)——Actor 
- [REINFORCE ](../chapter05_policy_gradient/pg-improvements)—— $G_t$  $G_t - V(s)$ 
  :::

##  REINFORCE  Actor-Critic

 5  REINFORCE （：[](../chapter05_policy_gradient/reinforce)）：

$$\nabla_\theta J \approx \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$$

$G_t$ —— REINFORCE 。 5 [](../chapter05_policy_gradient/pg-improvements)， $V(s)$ 。， episode ——[TD Error](./critic-training) $\delta = r + \gamma V(s') - V(s)$  $G_t - V(s)$ ：

$$\nabla_\theta J \approx \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \delta$$

：

|          | REINFORCE                 | Actor-Critic                                           |
| -------- | ------------------------- | ------------------------------------------------------ |
|  | $G_t$（MC，） | $\delta = r + \gamma V(s') - V(s)$（TD，） |
|  | episode             |                                                |
|      |                         |                                                      |
|      |                       | （[](../chapter03_mdp/dp-mc-td)）      |
|      |                         |  Critic                                        |

### ：

 CartPole ， $t$  $s_t$，""（$a_t = \text{right}$）， 5  episode 。：

|   |       |   |  $r$ |
| ----- | --------- | ----- | -------- |
| $t$   | $s_t$     | right | 1.0      |
| $t+1$ | $s_{t+1}$ | right | 1.0      |
| $t+2$ | $s_{t+2}$ | left  | 1.0      |
| $t+3$ | $s_{t+3}$ | right | 1.0      |
| $t+4$ | $s_{t+4}$ | right | 1.0      |

 $\gamma = 0.99$。

**REINFORCE 。** REINFORCE  episode 。 $t$  $G_t$：

$$
\begin{aligned}
G_t &= r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \gamma^3 r_{t+4} + \gamma^4 r_{t+5} \\
    &= 1.0 + 0.99 \times 1.0 + 0.99^2 \times 1.0 + 0.99^3 \times 1.0 + 0.99^4 \times 1.0 \\
    &= 1.0 + 0.99 + 0.9801 + 0.9703 + 0.9606 \\
    &= 4.9010.
\end{aligned}
$$

 $G_t$ 。 $\pi_\theta$  $s_t$  right  $\pi(\text{right}|s_t) = 0.6$，

$$
\log \pi(\text{right}|s_t) = \log 0.6 \approx -0.5108.
$$



$$
\nabla_\theta J \approx \nabla_\theta \log \pi(\text{right}|s_t) \cdot G_t = \nabla_\theta \log \pi(\text{right}|s_t) \times 4.9010.
$$

：，$G_t$  1.0（ 1 ）， 10.0（）。$G_t$ ， REINFORCE 。

> **REINFORCE **
>
> |                                       |                                                      |
> | ----------------------------------------- | -------------------------------------------------------- |
> | $\nabla_\theta \log \pi_\theta(a_t\|s_t)$ |  $\theta$ ， |
> | $G_t$                                     |  $t$  episode                  |
> | $r_{t+k}$                                 |  $t+k$                                 |
> | $\gamma$                                  | ，                         |

**Actor-Critic 。** Actor-Critic  episode 。 Critic  $V(s_t) = 2.0$，$V(s_{t+1}) = 3.0$。， $r_{t+1} = 1.0$， TD Error：

$$
\begin{aligned}
\delta &= r_{t+1} + \gamma V(s_{t+1}) - V(s_t) \\
       &= 1.0 + 0.99 \times 3.0 - 2.0 \\
       &= 1.0 + 2.97 - 2.0 \\
       &= 1.97.
\end{aligned}
$$

$\delta = 1.97 > 0$， Critic 。 TD Error ：

$$
\nabla_\theta J \approx \nabla_\theta \log \pi(\text{right}|s_t) \times 1.97.
$$

 $\log \pi(\text{right}|s_t) \approx -0.5108$， REINFORCE ，， TD Error。$\delta$  $G_t$——，。

> **Actor-Critic **
>
> |                                       |                                             |
> | ----------------------------------------- | ----------------------------------------------- |
> | $\nabla_\theta \log \pi_\theta(a_t\|s_t)$ |  $\theta$                 |
> | $\delta$                                  | TD Error， $A(s,a)$           |
> | $r_{t+1}$                                 |                             |
> | $\gamma V(s_{t+1})$                       | （Critic ） |
> | $V(s_t)$                                  | Critic （）         |

：

|        | REINFORCE                      | Actor-Critic                         |
| -------------- | ------------------------------ | ------------------------------------ |
|        | episode ，     | ， $r_{t+1}$  $s_{t+1}$  |
|        | $G_t = 4.9010$（5 ） | $\delta = 1.97$（ TD Error）     |
|        |            |                    |
|  |                              | Critic  $V(s_t)$  $V(s_{t+1})$ |
|    | （）           | （Critic ）          |

## Actor-Critic 

 Critic ，。Actor ，Critic ， $A(s,a)$ ：

```
Actor-Critic 

   s
    │
    ├──→ Actor（）
    │      π(a|s) →  a
    │                  │
    │               a
    │                  │
    │                  ▼
    │               →  r, s'
    │                  │
    ├──→ Critic（）  │
    │      V(s)  ──────────┤
    │      V(s') ──────────┤
    │                      │
    │      δ = r + γV(s') - V(s)
    │            │
    │            ▼
    │      Actor ：θ ← θ + α·∇log π(a|s)·δ
    │      Critic ：V(s) ← V(s) + α·δ
    │
    └──→ ，
```

（ $s$），：

|              |      |      |                  |          |
| ---------------- | -------- | -------- | -------------------- | ---------------- |
| Actor（）    |  |  $s$ |  $\pi(a\|s)$ |    |
| Critic（） |  |  $s$ |  $V(s)$      |  |

 Critic ，$V(s) \leftarrow V(s) + \alpha \cdot \delta$—— 3  [TD Learning](../chapter03_mdp/dp-mc-td) ？**Critic  3 [ $V(s)$](../chapter03_mdp/value-bellman)**，""。Actor [ $\pi(a|s)$](../chapter03_mdp/policy-objective) ， Critic 。

——Critic  Actor ""，Actor ， Critic 。 Actor-Critic 。

### 

 Actor-Critic 。 CartPole  $s = [0.05,\ 0.2,\ -0.03,\ 0.1]$。 $\theta$， Actor  Critic ：

|    |                  |           |
| ------ | -------------------- | ------------- |
| Actor  |  $\pi(a\|s)$ | $[0.7,\ 0.3]$ |
| Critic |  $V(s)$      | $1.5$         |

 $\pi(\text{left}|s) = 0.7$，$\pi(\text{right}|s) = 0.3$。

** 1 ：。**  $a = \text{right}$（ 2 ）。：

$$
\log \pi(\text{right}|s) = \log 0.3 \approx -1.2040.
$$

** 2 ：，。**  $r = 1.0$， $s' = [0.06,\ 0.25,\ -0.01,\ 0.08]$。

** 3 ：Critic 。**  $s'$  Critic（）：

$$
V(s') = 2.0.
$$

** 4 ： TD  TD Error。**

$$
\begin{aligned}
\text{TD } &= r + \gamma V(s') \\
               &= 1.0 + 0.99 \times 2.0 \\
               &= 1.0 + 1.98 \\
               &= 2.98.
\end{aligned}
$$

$$
\begin{aligned}
\delta &= \text{TD } - V(s) \\
       &= 2.98 - 1.5 \\
       &= 1.48.
\end{aligned}
$$

$\delta = 1.48 > 0$—— Critic ，" $s$  right"。

** 5 ： Actor Loss。**

$$
\begin{aligned}
L_{\text{actor}} &= -\log \pi(\text{right}|s) \cdot \delta \\
                 &= -(-1.2040) \times 1.48 \\
                 &= 1.2040 \times 1.48 \\
                 &= 1.7819.
\end{aligned}
$$

 $\delta$  `.detach()`—— Actor Loss， Critic 。

> **Actor Loss **
>
> |                |                                                                                        |
> | ------------------ | ------------------------------------------------------------------------------------------ |
> | $L_{\text{actor}}$ | Actor ，                                                 |
> | $\log \pi(a\|s)$   | ，$\theta$                                                     |
> | $\delta$           | TD Error，，** Actor **                                      |
> |                | ： $-\log\pi \cdot \delta$  $\log\pi \cdot \delta$ |

** 6 ： Critic Loss。**

$$
\begin{aligned}
L_{\text{critic}} &= \delta^2 \\
                  &= 1.48^2 \\
                  &= 2.1904.
\end{aligned}
$$

—— $V(s)$  TD  $r + \gamma V(s')$。

> **Critic Loss **
>
> |                                |                                                |
> | ---------------------------------- | -------------------------------------------------- |
> | $L_{\text{critic}}$                | Critic ， $V(s)$  TD         |
> | $\delta = r + \gamma V(s') - V(s)$ | TD Error， $V(s)$  Critic        |
> | $\delta^2$                         | ， |

** 7 ：。**

$$
\begin{aligned}
L_{\text{total}} &= L_{\text{actor}} + L_{\text{critic}} \\
                 &= 1.7819 + 2.1904 \\
                 &= 3.9723.
\end{aligned}
$$

，：

- **Actor **：$\nabla_\theta L_{\text{actor}} = -\nabla_\theta \log \pi(\text{right}|s) \cdot 1.48$。$\delta$ ，——$\delta > 0$  right ，$\delta < 0$ 。
- **Critic **：$\nabla_\theta L_{\text{critic}} = 2\delta \cdot \nabla_\theta V(s) = 2 \times 1.48 \cdot \nabla_\theta V(s) = 2.96 \cdot \nabla_\theta V(s)$。$V(s)$  $\theta$ ， Critic  TD 。

：

|  |                |                                       |                        |
| ---- | ------------------ | ----------------------------------------- | -------------------------- |
|  | $s$                | $\text{Actor}(s),\ \text{Critic}(s)$      | $\pi=[0.7,0.3],\ V(s)=1.5$ |
|  | $\pi$              | $\text{Categorical}(\pi).\text{sample}()$ | $a=\text{right}$           |
|  | $s,\ a$            | $\text{env.step}(a)$                      | $r=1.0,\ s'$               |
|  | $s'$               | $\text{Critic}(s')$                       | $V(s')=2.0$                |
| TD   | $r,\ V(s'),\ V(s)$ | $r+\gamma V(s')-V(s)$                     | $\delta=1.48$              |
|  | $\log\pi,\ \delta$ | $-\log\pi\cdot\delta + \delta^2$          | $L=3.9723$                 |

###  PyTorch  Actor-Critic

Actor-Critic  REINFORCE  Critic ，：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np

# ==========================================
# 1. Actor-Critic （）
# ==========================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # 
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
        )
        # Actor ：
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        # Critic ：
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        features = self.shared(x)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value

# ==========================================
# 2. （， episode ）
# ==========================================
env = gym.make("CartPole-v1")
model = ActorCritic(state_dim=4, action_dim=2)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
gamma = 0.99

reward_history = []

for episode in range(500):
    state, _ = env.reset()
    total_reward = 0

    while True:
        state_t = torch.FloatTensor(state)

        # Actor ，Critic 
        probs, value = model(state_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # 
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        total_reward += reward

        # Critic 
        with torch.no_grad():
            _, next_value = model(torch.FloatTensor(next_state))
            next_value = 0 if done else next_value

        # TD Error = （： 6.1  A ≈ δ）
        td_target = reward + gamma * next_value
        td_error = td_target - value

        # Actor ： × 
        actor_loss = -log_prob * td_error.detach()

        # Critic ： V(s)  TD Target（： 6.2  L = δ²）
        critic_loss = td_error.pow(2)

        # 
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        if done:
            break

    reward_history.append(total_reward)
    if (episode + 1) % 50 == 0:
        avg = np.mean(reward_history[-50:])
        print(f"Episode {episode+1} | Avg Reward: {avg:.1f}")
```

 5  REINFORCE ，： Critic （ $V(s)$）， TD Error（`td_target - value`） $G_t$，Critic （MSE）， episode 。

### ：

，。 $s = [0.1,\ 0.2,\ -0.3,\ 0.4]$， $\gamma = 0.99$。

**。**  `state_t = torch.FloatTensor([0.1, 0.2, -0.3, 0.4])` ：

```python
probs, value = model(state_t)
# probs = tensor([0.6000, 0.4000])   ← Actor ：left  0.6, right  0.4
# value = tensor(1.2000)             ← Critic ：V(s) = 1.2
```

**。**

```python
dist = torch.distributions.Categorical(probs)
action = dist.sample()           # action = tensor(1)， right
log_prob = dist.log_prob(action) # log_prob = log(0.4) = tensor(-0.9163)
```

$\log \pi(\text{right}|s) = \log 0.4 \approx -0.9163$。

**。**  `action.item() = 1`（right）：

```python
next_state, reward, terminated, truncated, _ = env.step(action.item())
# reward = 1.0
# terminated = False, truncated = False
```

**。**

```python
with torch.no_grad():
    _, next_value = model(torch.FloatTensor(next_state))
    # next_value = tensor(2.0000)    ← V(s') = 2.0
    # done = False,  next_value 
```

** TD  TD Error。**

$$
\text{td\_target} = r + \gamma \cdot V(s') = 1.0 + 0.99 \times 2.0 = 2.98.
$$

$$
\text{td\_error} = \text{td\_target} - V(s) = 2.98 - 1.2 = 1.78.
$$

```python
td_target = reward + gamma * next_value  # = 1.0 + 0.99 * 2.0 = tensor(2.9800)
td_error  = td_target - value            # = 2.98 - 1.2 = tensor(1.7800)
```

**。**

Actor Loss（$\delta$  `.detach()` ，）：

$$
L_{\text{actor}} = -\log\pi(\text{right}|s) \cdot \delta = -(-0.9163) \times 1.78 = 1.6310.
$$

```python
actor_loss = -log_prob * td_error.detach()  # = -(-0.9163) * 1.78 = tensor(1.6310)
```

Critic Loss（$\delta$  $V(s)$， $V(s)$  Critic ）：

$$
L_{\text{critic}} = \delta^2 = 1.78^2 = 3.1684.
$$

```python
critic_loss = td_error.pow(2)  # = 1.78^2 = tensor(3.1684)
```

**。**

$$
L = L_{\text{actor}} + L_{\text{critic}} = 1.6310 + 3.1684 = 4.7994.
$$

```python
loss = actor_loss + critic_loss  # = tensor(4.7994)
```

**。** `loss.backward()` ，`optimizer.step()`  $\alpha = 0.001$ 。：

- **Actor **：$\delta = 1.78 > 0$， right 。 $\pi(\text{right}|s)$—— right。
- **Critic **：$V(s) = 1.2$  TD  $2.98$。$\delta^2$  $V(s)$， $r + \gamma V(s')$。

：

|           |          |                                    |
| ------------- | ---------- | -------------------------------------- |
| `probs`       | [0.6, 0.4] | Actor              |
| `value`       | 1.2        | Critic                 |
| `log_prob`    | -0.9163    |  right               |
| `reward`      | 1.0        |                      |
| `next_value`  | 2.0        | Critic                 |
| `td_target`   | 2.98       | $r + \gamma V(s')$                     |
| `td_error`    | 1.78       | $\delta = \text{td\textunderscore{}target} - V(s)$ |
| `actor_loss`  | 1.6310     | $-\log\pi \cdot \delta$（.detach ）  |
| `critic_loss` | 3.1684     | $\delta^2$                             |
| `loss`        | 4.7994     | $L_{\text{actor}} + L_{\text{critic}}$ |

### CartPole  Actor-Critic 

```
Actor-Critic  CartPole 

 500 ┤
     │                              ━━━━━━━━━━━━━━━
 400 ┤                         ━━━━
     │                    ━━━━
 300 ┤              ━━━━━
     │         ━━━━
 200 ┤    ━━━━
     │ ━━
 100 ┤╱
     └────────────────────────────────────────────
     0    50   100  150  200  250  300  350  400  450  500
                    Episode

  REINFORCE （、）
```

Actor-Critic  CartPole  200-300  episode  500 （）， REINFORCE  500+ episode 。""——，。

## Actor-Critic 

Actor-Critic ，。：

|                                                                |               |                                           |
| ------------------------------------------------------------------ | ----------------- | ------------------------------------------------- |
| [ 7  PPO](../chapter07_ppo/intro)                              | PPO-Clip          | ，""                |
| [ 7  GAE](../chapter07_ppo/gae-reward-model)                   |       |  TD Error ，- |
| [ 9  DPO](../chapter09_alignment/intro)                        |  Actor-Critic |  Critic， on-policy       |
| [ 9  GRPO](../chapter09_grpo_rlvr/grpo-practice-and-mechanism) |  Critic       |  $V(s)$，               |

： + 。""""。

<details>
<summary>： Actor-Critic  REINFORCE ， Critic（ V）？</summary>

 Critic 。Critic  $V(s)$  $Q(s,a)$， $\arg\max_a Q(s,a)$（：[](../chapter03_mdp/value-q)）——， $\arg\max$ （）。

Actor ：，。——Critic ""，Actor ""，。

</details>

<details>
<summary>：Actor-Critic ""？？</summary>

 Critic [（Bootstrapping）](../chapter03_mdp/dp-mc-td)——Critic  $V(s')$  $V(s)$。 $V(s')$ ，。——。

。， REINFORCE 。 7  GAE "-"—— $\lambda$  TD（） MC（）。

</details>

 Actor-Critic ——[Actor-Critic ](./ac-frontier)。

---

[^2]: Sutton, R. S., et al. (1999). Policy gradient methods for reinforcement learning with function approximation. _Advances in Neural Information Processing Systems_, 12.
