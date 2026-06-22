# 5.2  REINFORCE

 Policy-Based ：DQN  $\arg\max$ ， $\pi_\theta(a|s)$ 。：""？？

## 

 3 [](../chapter03_mdp/policy-objective) $J(\theta)$——""。：， $\pi_\theta$ [](../chapter03_mdp/mdp)。

$$J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]$$

|                       |      |                                                  |
| ------------------------- | -------- | ---------------------------------------------------- |
| $\theta$                  |  | ——               |
| $\pi_\theta$              |  | ，                     |
| $J(\theta)$               |  | ""—— $\theta$  |
| $\mathbb{E}_{\pi\theta}$  |      |  $\pi_\theta$ ，           |
| $\gamma^t r_t$            |  |  $t$ ，""            |

$J(\theta)$ ——： $J(\theta)$  $\theta$。

### ： 3  episode  $J(\theta)$

 episode  3 ， $r_0=1$，$r_1=2$，$r_2=3$， $\gamma=0.9$。

$$
\begin{aligned}
\sum_{t=0}^{2} \gamma^t r_t &= \gamma^0 r_0 + \gamma^1 r_1 + \gamma^2 r_2 \\
&= 0.9^0 \times 1 + 0.9^1 \times 2 + 0.9^2 \times 3 \\
&= 1 \times 1 + 0.9 \times 2 + 0.81 \times 3 \\
&= 1 + 1.8 + 2.43 \\
&= 5.23.
\end{aligned}
$$

****。$J(\theta)$ —— $\pi_\theta$ ，。 $\pi_\theta$ ， $J(\theta)$ 。，：

|  |  | $J(\theta)$ |
| ---- | ------------------ | ----------- |
|    |  $5.23$  |         |
|    |  $2.10$  |         |

 $J(\theta)$ （） $\theta$。

## 

 $J(\theta)$ ？：。

$$\theta \leftarrow \theta + \alpha \, \nabla_\theta J(\theta)$$

|                       |      |                                        |
| ------------------------- | -------- | ------------------------------------------ |
| $\nabla_\theta J(\theta)$ |      | "，" |
| $\alpha$                  |    | ""——，       |
| $+$                       |  | ——，       |

### ：

 $\theta = [0.5,\ 0.3,\ -0.1]$， $\nabla_\theta J(\theta) = [0.1,\ -0.2,\ 0.05]$， $\alpha = 0.01$。：

$$
\begin{aligned}
\theta_0 &\leftarrow 0.5 + 0.01 \times 0.1 = 0.5 + 0.001 = 0.501, \\
\theta_1 &\leftarrow 0.3 + 0.01 \times (-0.2) = 0.3 - 0.002 = 0.298, \\
\theta_2 &\leftarrow -0.1 + 0.01 \times 0.05 = -0.1 + 0.0005 = -0.0995.
\end{aligned}
$$

 $\theta = [0.501,\ 0.298,\ -0.0995]$。。$\alpha$ ：$\alpha=0.01$ ，。

 $\nabla_\theta J(\theta)$ ？ $\mathbb{E}$——。，。——， 100 。 100 ，****。：，， $\nabla_\theta J(\theta)$。

## 

。1992 ，Williams  REINFORCE ： $\nabla_\theta J(\theta)$， [^1]。 Sutton  2000  [^2]。

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t \right]$$

：

|                                         |            |                                                       |
| ------------------------------------------- | -------------- | --------------------------------------------------------- |
| $\nabla_\theta$                             |          | ""                                            |
| $\log \pi_\theta(a_t \| s_t)$               |        |  $s_t$ ， $a_t$             |
| $\nabla_\theta \log \pi_\theta(a_t \| s_t)$ |  | ""                    |
| $G_t$                                       |        |  $t$ ——"" |
|  $\mathbb{E}$                           |            | ""——                            |

：**（$G_t$ ），；（$G_t$ ），。**

### ：

。：A  30%，B  70%。 $\theta$（ $\theta$ ， B ）， $\pi_\theta(B|s) = 0.7$，$\pi_\theta(A|s) = 0.3$。

** 1： B， = 1.0**

，：

$$
\log \pi_\theta(B|s) = \log(0.7) = -0.357.
$$

，。 $\nabla_\theta \log \pi_\theta(B|s) = 0.5$（，）。

，。， $G_0 = r_0 = 1.0$：

$$
\nabla_\theta \log \pi_\theta(B|s) \cdot G_0 = 0.5 \times 1.0 = 0.5.
$$

， $\theta \leftarrow \theta + \alpha \times 0.5$。，$\pi_\theta(B|s)$ ——" B ， B"。

** 2： A， = 0**

$$
\log \pi_\theta(A|s) = \log(0.3) = -1.204.
$$

 $\nabla_\theta \log \pi_\theta(A|s) = -0.5$（ A  B ）。 $G_0 = 0$：

$$
\nabla_\theta \log \pi_\theta(A|s) \cdot G_0 = (-0.5) \times 0 = 0.
$$

，。 A  0，。

** 3： A， = 0.5**

$$
\nabla_\theta \log \pi_\theta(A|s) \cdot G_0 = (-0.5) \times 0.5 = -0.25.
$$

， $\theta \leftarrow \theta + \alpha \times (-0.25)$。，$\pi_\theta(B|s)$ 、$\pi_\theta(A|s)$ ——"A ， A"。 A  B， B  A， B。

### ：3  episode 

 CartPole ：3  episode，$\gamma=0.9$。

|  |   |        |       |
| ---- | ----- | ---------- | --------- |
| 0    | $s_0$ | $a_0$ =  | $r_0 = 1$ |
| 1    | $s_1$ | $a_1$ =  | $r_1 = 2$ |
| 2    | $s_2$ | $a_2$ =  | $r_2 = 3$ |

 $G_t$。$G_2$ ：

$$
G_2 = r_2 = 3.
$$

$G_1$  1 ：

$$
G_1 = r_1 + \gamma r_2 = 2 + 0.9 \times 3 = 2 + 2.7 = 4.7.
$$

$G_0$  0 ：

$$
G_0 = r_0 + \gamma r_1 + \gamma^2 r_2 = 1 + 0.9 \times 2 + 0.81 \times 3 = 1 + 1.8 + 2.43 = 5.23.
$$

：

|  | $\log \pi_\theta(a_t \| s_t)$               | $\nabla_\theta \log \pi_\theta(a_t \| s_t)$ | $G_t$  |
| ---- | ------------------------------------------- | ------------------------------------------- | ------ |
| 0    | $\log \pi_\theta(\text{}\mid s_0) = -0.4$ | $[0.3, -0.1]$                               | $5.23$ |
| 1    | $\log \pi_\theta(\text{}\mid s_1) = -0.7$ | $[-0.2, 0.4]$                               | $4.7$  |
| 2    | $\log \pi_\theta(\text{}\mid s_2) = -0.3$ | $[0.1, 0.2]$                                | $3$    |

：

$$
\begin{aligned}
\text{ 0：} \quad \nabla_\theta \log \pi_\theta(a_0|s_0) \cdot G_0 &= [0.3, -0.1] \times 5.23 = [1.569, -0.523], \\
\text{ 1：} \quad \nabla_\theta \log \pi_\theta(a_1|s_1) \cdot G_1 &= [-0.2, 0.4] \times 4.7 = [-0.94, 1.88], \\
\text{ 2：} \quad \nabla_\theta \log \pi_\theta(a_2|s_2) \cdot G_2 &= [0.1, 0.2] \times 3 = [0.3, 0.6].
\end{aligned}
$$

：

$$
\begin{aligned}
\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t &= [1.569, -0.523] + [-0.94, 1.88] + [0.3, 0.6] \\
&= [1.569 - 0.94 + 0.3,\ -0.523 + 1.88 + 0.6] \\
&= [0.929,\ 1.957].
\end{aligned}
$$

 $\alpha = 0.01$， $\theta \leftarrow \theta + 0.01 \times [0.929,\ 1.957]$。（$G_1=4.7$  $0.4$ ），" $s_1$ "，" $s_1$ "。

### 

 $\nabla_\theta \pi_\theta(a_t|s_t) \cdot G_t$， $\log$？

，****（Log-Derivative Trick）。：

$$\nabla_\theta \log \pi = \frac{\nabla_\theta \pi}{\pi}$$

" $\pi$" $\pi$ ，。， $\pi$  $(0, 1)$ ，，。$\log$  $(0, 1)$  $(-\infty, 0)$，。

### ：

：$\pi_\theta(a|s) = 0.7$， $\pi_\theta(a|s)$  $0.71$，

$$
\nabla_\theta \pi_\theta(a|s) \approx 0.71 - 0.7 = 0.01.
$$

：

$$
\nabla_\theta \pi_\theta(a|s) = 0.01.
$$

：

$$
\nabla_\theta \log \pi_\theta(a|s) = \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)} = \frac{0.01}{0.7} = 0.0143.
$$

 $\pi_\theta(a'|s) = 0.05$， $0.06$：

$$
\nabla_\theta \pi_\theta(a'|s) = 0.01, \quad \nabla_\theta \log \pi_\theta(a'|s) = \frac{0.01}{0.05} = 0.2.
$$

（ $0.01$）， $0.2/0.0143 \approx 14$ 。： $5\%$  $6\%$， $20\%$； $70\%$  $71\%$， $1.4\%$。 $\pi$ ，。

<details>
<summary>：</summary>

：

$$\nabla_\theta J(\theta) = \nabla_\theta \sum_{\tau} P(\tau; \theta) \sum_t r_t(\tau)$$

 $\tau = (s_0, a_0, s_1, a_1, \ldots)$ ，$P(\tau; \theta)$  $\tau$ 。 $P(\tau; \theta)$（）：

$$\nabla_\theta J(\theta) = \sum_{\tau} \nabla_\theta P(\tau; \theta) \sum_t r_t(\tau)$$

： $\nabla_\theta P = P \cdot \nabla_\theta \log P$：

$$\nabla_\theta J(\theta) = \sum_{\tau} P(\tau; \theta) \left( \nabla_\theta \log P(\tau; \theta) \right) \sum_t r_t(\tau)$$

：$P(\tau; \theta) = \prod_t \pi_\theta(a_t|s_t) \cdot P(s_{t+1}|s_t, a_t)$。 $\theta$ ， $P(s'|s,a)$  $\theta$，：

$$\nabla_\theta \log P(\tau; \theta) = \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)$$

，。：（）。****——。

</details>

## REINFORCE 

。**REINFORCE** ——[](../chapter03_mdp/dp-mc-td)。：

1.  $\pi_\theta$  episode，、
2. ， episode  $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$
3. ：$\nabla_\theta J \approx \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$
4. ：$\theta \leftarrow \theta + \alpha \nabla_\theta J$

 PyTorch ，：

```python
loss = -log_prob * G_t  #  PyTorch （），（）
```

：

```python
# REINFORCE （）
for t in range(len(rewards)):
    G_t = sum(gamma ** k * rewards[t + k] for k in range(len(rewards) - t))
    loss += -log_probs[t] * G_t

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### ： episode  REINFORCE 

 3  episode，$\gamma=0.9$：

|  |   |  |     | $\log \pi_\theta(a_t \| s_t)$ |
| ---- | ----- | ---- | ------- | ----------------------------- |
| 0    | $s_0$ |    | $r_0=1$ | $-0.4$                        |
| 1    | $s_1$ |    | $r_1=2$ | $-0.7$                        |
| 2    | $s_2$ |    | $r_2=3$ | $-0.3$                        |

**： $G_t$。**

$G_2$  2 （ 1 ）：

$$
G_2 = r_2 = 3.
$$

$G_1$  1 （2 ）：

$$
G_1 = r_1 + \gamma r_2 = 2 + 0.9 \times 3 = 2 + 2.7 = 4.7.
$$

$G_0$  0 （3 ）：

$$
G_0 = r_0 + \gamma r_1 + \gamma^2 r_2 = 1 + 0.9 \times 2 + 0.81 \times 3 = 1 + 1.8 + 2.43 = 5.23.
$$

：

|  | $r_t$ | $G_t$ |
| ---- | ----- | ----- |
| 0    | 1     | 5.23  |
| 1    | 2     | 4.7   |
| 2    | 3     | 3     |

**：。**

 $\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$。 PyTorch ，`log_prob`  $\log \pi_\theta(a_t|s_t)$，。 loss ：

$$
\begin{aligned}
\text{loss} &= -\sum_{t=0}^{2} \log \pi_\theta(a_t|s_t) \cdot G_t \\
&= -(\log \pi_\theta(\text{}|s_0) \cdot G_0 + \log \pi_\theta(\text{}|s_1) \cdot G_1 + \log \pi_\theta(\text{}|s_2) \cdot G_2) \\
&= -((-0.4) \times 5.23 + (-0.7) \times 4.7 + (-0.3) \times 3) \\
&= -(−2.092 + (−3.29) + (−0.9)) \\
&= -(-6.282) \\
&= 6.282.
\end{aligned}
$$

**：。**

`loss.backward()`  $\nabla_\theta \text{loss}$。 $\text{loss} = -\sum_t \log \pi_\theta(a_t|s_t) \cdot G_t$，

$$
\nabla_\theta \text{loss} = -\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t.
$$

`optimizer.step()`  $\theta \leftarrow \theta - \alpha \cdot \nabla_\theta \text{loss}$（PyTorch ），：

$$
\theta \leftarrow \theta - \alpha \cdot \left(-\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right) = \theta + \alpha \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t.
$$

。""。

### ：

 CartPole ， `loss = -log_prob * reward` 。

，：A  30%，B  70%。 Softmax ， A  B 。：

```python
probs = policy(state)
dist = torch.distributions.Categorical(probs)
action = dist.sample()          # 
log_prob = dist.log_prob(action)  # log π(a|s)

reward = pull_arm(action.item())  # 

loss = -log_prob * reward         # REINFORCE 
```

 300  episode ， B  0.5  0.85–0.95——""。，。 0.01  0.1， A  B 。

 REINFORCE ——****。

## REINFORCE 

$G_t$  $t$  episode ——。， $G_t$：

|    |          | $G_t$ |
| ------ | ---------------------- | ----- |
|  |  |   |
|  |  |   |

 $G_t$ ""—— $G_t$ ，**，**。——，。

### ：

：$\pi_\theta(B|s) = 0.7$，$\pi_\theta(A|s) = 0.3$。 B（），。

**Episode 1： B， 1.0（）**

$$
\text{} = \nabla_\theta \log \pi_\theta(B|s) \times 1.0.
$$

 $\pi_\theta(B|s)$。

**Episode 2： B， 0.0（，）**

$$
\text{} = \nabla_\theta \log \pi_\theta(B|s) \times 0.0 = 0.
$$

，。

。 3  episode ，$\gamma=0.9$， $s_0$ ""：

| Episode | $r_0$ | $r_1$ | $r_2$ | $G_0$                   |  |
| ------- | ----- | ----- | ----- | ----------------------- | ------------ |
| 1       | 1     | 2     | 3     | $1 + 1.8 + 2.43 = 5.23$ |        |
| 2       | 1     | 0     | 0     | $1 + 0 + 0 = 1$         |        |

 $s_0$ ""，$G_0$  4 。Episode 1  $\pi(\text{}|s_0)$ ，Episode 2 。 $G_0$ " $s_0$ "， $r_1$  $r_2$ ——， $G_0$。

，。（ CartPole），——，。

## 

（ A  B、CartPole /），：

|                   |               |                                  |
| ----------------- | ------------------------- | -------------------------------------------- |
|               | CartPole /、LLM   | 、                   |
|             | Softmax（） | （ $\mu$  $\sigma$） |
|           |  Softmax      |  $\mathcal{N}(\mu, \sigma^2)$          |
| $\log \pi$  | `log_softmax`             |                        |

，"/"""。 Value-Based ：DQN  $\arg\max$ ，，。

<details>
<summary>：REINFORCE  Q-Learning ？</summary>

Q-Learning  $Q(s,a)$（""）， $\arg\max Q$ 。REINFORCE  $\theta$， Q 。

：Q-Learning  off-policy （），REINFORCE  on-policy （）；Q-Learning （ max），REINFORCE （）。

</details>

REINFORCE ，。：""，，。 5.4 。 CartPole  REINFORCE：[：CartPole ](./cartpole)。

---

[^1]: Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. _Machine Learning_, 8(3-4), 229-256. [DOI](https://doi.org/10.1007/BF00992696)

[^2]: Sutton, R. S., et al. (1999). Policy gradient methods for reinforcement learning with function approximation. _Advances in Neural Information Processing Systems_, 12.
