# 5.2  REINFORCE

，""" B"。：`loss = -log_prob * reward`。

：？`log_prob`  `reward`，""？——" $Q$ "（：[Q(s,a)：](../chapter03_mdp/value-q)）。

。，——****。 Policy-Based RL ， PPO、GRPO 。

## Value-Based  Policy-Based

，， 4  DQN 。——、、。

### 

**Value-Based**（ 4  DQN）****： $Q(s,a)$，，。——""， $\arg\max_a Q(s,a)$ （：[Q(s,a) ](../chapter03_mdp/value-q)）。

**Policy-Based**（）****： $\pi_\theta(a|s)$，， $\theta$ （：[ $J(\theta)$](../chapter03_mdp/policy-objective)）。——。

|            | Value-Based（DQN）                                                                  | Policy-Based（）                 |
| ---------- | ----------------------------------------------------------------------------------- | ---------------------------------------- |
|      | $Q(s,a)$：                                                          | $\pi_\theta(a\|s)$： |
|  | $\arg\max_a Q(s,a)$（）                                                     |  $\pi_\theta(\cdot\|s)$          |
|    | （）                                                              | （）                   |
|    | [](../chapter03_mdp/value-bellman) + [TD ](../chapter03_mdp/dp-mc-td) |  +                   |

### 

Value-Based ：****。$\arg\max$  $Q$ ——CartPole  2 ，； $[-10, 10]^6$，$\arg\max$ 。—— token ， $Q$ 。

Policy-Based 。，—— Softmax，。，"/"""。

### 

DQN （ $\arg\max$），—— 4  $\varepsilon$-greedy： $\varepsilon$ ， $Q$ 。$\varepsilon$ ，，（：[DQN ](../chapter04_dqn/dqn-components)）。

，—— 30% ， 30% 。。，，" B"—— $\varepsilon$ ，。

### 

。DQN  **off-policy** ：，（：[](../chapter04_dqn/dqn-components)）。 batch ，。

 **on-policy** ： $\mathbb{E}_{\pi_\theta}$  $\pi_\theta$ 。，。 DQN——。

### 

|          | Value-Based                | Policy-Based                       |
| -------- | -------------------------- | ---------------------------------- |
|  |                      |  +                         |
|  |                      | （）                 |
|  | Off-policy（） | On-policy（）          |
|      | （TD ）          | （）           |
|  | DQN（ 4 ）             | REINFORCE（） → PPO（ 7 ） |

。，—— 6  Actor-Critic ：，。， Policy-Based 。

## 

，：""？

 3  MDP ，[](../chapter03_mdp/value-bellman) $V^\pi(s)$——" $s$ ， $\pi$，"。""""——""，""。 3 [](../chapter03_mdp/policy-objective) $J(\theta)$ 。：， $\pi_\theta$ [](../chapter03_mdp/mdp)。

$$J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]$$

""：

|                       |      |                                                |
| ------------------------- | -------- | ---------------------------------------------------- |
| $\theta$                  |  | ——               |
| $\pi_\theta$              |  | ，                     |
| $J(\theta)$               |  | ""—— $\theta$  |
| $\mathbb{E}_{\pi_\theta}$ |      |  $\pi_\theta$ ，           |
| $\gamma^t r_t$            |  |  $t$ ，""            |

$J(\theta)$ ——： $J(\theta)$  $\theta$。" $\theta$"。

## 

 $J(\theta)$ ？：。

$$\theta \leftarrow \theta + \alpha \, \nabla_\theta J(\theta)$$

|                       |      |                                        |
| ------------------------- | -------- | -------------------------------------------- |
| $\nabla_\theta J(\theta)$ |      | "，"   |
| $\alpha$                  |    | ""——，       |
| $+$                       |  | ——， |

：$\nabla_\theta J(\theta)$ ？

 $J(\theta)$  $\mathbb{E}$——。，，。——， 100 。

## 

。1992 ，Ronald Williams  REINFORCE ： $\nabla_\theta J(\theta)$， [^1]。 Sutton  2000  [^2]。

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t \right]$$

，：

|                                         |            |                                                     |
| ------------------------------------------- | -------------- | --------------------------------------------------------- |
| $\nabla_\theta$                             |          | ""                                            |
| $\log \pi_\theta(a_t \| s_t)$               |        |  $s_t$ ， $a_t$             |
| $\nabla_\theta \log \pi_\theta(a_t \| s_t)$ |  | ""                    |
| $G_t$                                       |        |  $t$ ——"" |
|  $\mathbb{E}$                           |            | ""——                            |

：**（$G_t$ ），；（$G_t$ ），。**

—— B （$G_t$ ）， B ； A ，，。。

### 

： $\nabla_\theta \pi_\theta(a_t|s_t) \cdot G_t$， $\log$？

，****（Log-Derivative Trick）。：

$$\nabla_\theta \log \pi = \frac{\nabla_\theta \pi}{\pi}$$

" $\pi$" $\pi$ ，。， $\pi$  $(0, 1)$ ，，。$\log$  $(0, 1)$  $(-\infty, 0)$，、。

<details>
<summary>：</summary>

：

$$\nabla_\theta J(\theta) = \nabla_\theta \sum_{\tau} P(\tau; \theta) \sum_t r_t(\tau)$$

 $\tau = (s_0, a_0, s_1, a_1, \ldots)$ ，$P(\tau; \theta)$  $\tau$ 。 $P(\tau; \theta)$（）：

$$\nabla_\theta J(\theta) = \sum_{\tau} \nabla_\theta P(\tau; \theta) \sum_t r_t(\tau)$$

： $\nabla_\theta P = P \cdot \nabla_\theta \log P$（ $P$  $P$）：

$$\nabla_\theta J(\theta) = \sum_{\tau} P(\tau; \theta) \left( \nabla_\theta \log P(\tau; \theta) \right) \sum_t r_t(\tau)$$

：$P(\tau; \theta) = \prod_t \pi_\theta(a_t|s_t) \cdot P(s_{t+1}|s_t, a_t)$。 $\theta$ ， $P(s'|s,a)$  $\theta$，：

$$\nabla_\theta \log P(\tau; \theta) = \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)$$

，。：（）。****——。

</details>

## REINFORCE 

。**REINFORCE** ——[](../chapter03_mdp/dp-mc-td)（：MC ""）。：

1.  $\pi_\theta$  episode，、
2. ， episode  $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$
3. ：$\nabla_\theta J \approx \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$
4. ：$\theta \leftarrow \theta + \alpha \nabla_\theta J$

 PyTorch ，：

```python
loss = -log_prob * G_t  #  PyTorch （），（）
```

，`loss = -log_prob * reward`  REINFORCE （$G_t = r_t$，，）。

```python
# REINFORCE （）
for t in range(len(rewards)):
    G_t = sum(gamma ** k * rewards[t + k] for k in range(len(rewards) - t))
    loss += -log_probs[t] * G_t

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## REINFORCE 

REINFORCE ，""——****。

？ $G_t$  $t$  episode ——。， $G_t$：

|    |          | $G_t$ |
| ------ | ---------------------- | ----- |
|  |  |   |
|  |  |   |

， $G_t$ ""—— $G_t$ ，**，**。——，。

——，。（ 0.01  0.1）， A  B ，。。

## 

（ A  B），。——""：

|                   |               |                                  |
| ----------------- | ------------------------- | -------------------------------------------- |
|               | CartPole /、LLM   | 、                   |
|             | Softmax（） | （ $\mu$  $\sigma$） |
|           |  Softmax      |  $\mathcal{N}(\mu, \sigma^2)$          |
| $\log \pi$  | `log_softmax`             |                        |

—— PPO ， LLM （ token ）（）。 Value-Based ：DQN  $\arg\max$ （），，。

<details>
<summary>：REINFORCE  Q-Learning ？</summary>

Q-Learning  $Q(s,a)$（""）， $\arg\max Q$ ——，。REINFORCE  $\theta$， Q ——""，""。

：Q-Learning  off-policy （），REINFORCE  on-policy （）；Q-Learning （ max），REINFORCE （）。

</details>

<details>
<summary>： REINFORCE  episode ？</summary>

 $G_t$  $t$  episode 。， $G_t$ 。，——。

： $G_t$， episode 。""？ 3 [](../chapter03_mdp/value-bellman)—— $V(s)$ ，""""。 Actor-Critic 。

</details>

REINFORCE ，""。：""，，。 Actor-Critic 。——[Actor-Critic ](../chapter06_actor_critic/actor-critic)——。

---

[^1]: Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. _Machine Learning_, 8(3-4), 229-256. [DOI](https://doi.org/10.1007/BF00992696)

[^2]: Sutton, R. S., et al. (1999). Policy gradient methods for reinforcement learning with function approximation. _Advances in Neural Information Processing Systems_, 12.
