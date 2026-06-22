# 3.2 MDP：

## 

****

-  MDP 、、、。
-  $G_t$  RL 。
-  $a=\pi(s)$  $\pi(a\mid s)$。

， RL ：****，****，**** RL 。，——，。”””-”。

。CartPole ，；，；，。””，”****，”——，。””””。

，——**（Markov Decision Process, MDP）**。，””：$\mathcal{S}$ ，$\mathcal{A}$ ，$P$ ，$R$ ，$\gamma$ 。

****——MDP 、$G_t$、$\pi$ ，” RL ”。——**** $V(s)$ ****。

，：****（MDP ）、****（ $G_t$）、****（ $\pi$）。

::: info 
MDP ****。 RL —— Q-learning  PPO， RLHF—— MDP 。 MDP （、）， CartPole、Atari、LLM  MDP。
:::

****

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma) \quad \text{（MDP ：）}
$$

> ** (Markov Decision Process)：**
>
> - $\mathcal{S}$：（State Space），。
> - $\mathcal{A}$：（Action Space），。
> - $P$：（Transition Probability），。
> - $R$：（Reward Function），。
> - $\gamma$：（Discount Factor），””””。

$$
G_t = \sum_{k=0}^{\infty}\gamma^k r_{t+k} = r_t + \gamma G_{t+1} \quad \text{（：）}
$$

> ** (Discounted Return)：**
>
> - $G_t$： $t$ ，（Return）。
> - $r_{t+k}$： $t+k$ 。

$$
a = \pi(s), \qquad \pi(a\mid s) = P(A_t=a\mid S_t=s) \quad \text{（：）}
$$

> ** (Policy)：**
>
> - $\pi$：（Policy），。
> - $\pi(s)$：， $s$  $a$。
> - $\pi(a\mid s)$：， $s$  $a$ 。

## MDP ：、

 MDP ， $(S, A, P, R, \gamma)$。，：**（S）（A），（P），（R），，—— $\gamma$ ""**。

 RL ，：

- ****（S）—— ？""？
- ****（A）—— ？（），（、）？
- ****（P）—— ，？？
- ****（R）—— ，？，？
- ****（γ）—— ？ 1 ？

——。 S，； A，； P，； R，""； γ，。**""。**

|  |                    |      |          |              | CartPole         |
| ---- | ---------------------- | -------- | ------------ | ------------------ | ---------------- |
| S    | State                  |  | ""     | {}（）   | R⁴（4 ） |
| A    | Action                 |  | ""     | { A,  B}       | {, }     |
| P    | Transition Probability |  | ""     |    |    |
| R    | Reward                 |  | ""     |  +1， -1   |  +1          |
| γ    | Discount Factor        |  | "" | （） |  0.99        |

###  S

。——。CartPole  4 ：

$$s = [x,\ \dot{x},\ \theta,\ \dot{\theta}]$$

、、。

MDP ****：，，。，，。

""—— CartPole """"，。，。**。**

###  A

 { A,  B}，CartPole  {, }， $a \in \mathbb{R}^n$。

：

|             |                      |            |
| ------------------- | ------------------------ | ------------------ |
| （ {, }） | ， | Q-learning / DQN   |
| （）  |                  | （PPO ） |

###  P

$$P(s' \mid s, a) = \text{ } s \text{  } a \text{ ， } s' \text{ }$$

，，。CartPole ，$P$ 。 $\theta = 0.05$（），""：

$$P(\theta' = 0.03 \mid \theta = 0.05, \text{}) \approx 0.7 \qquad (\text{，})$$

$$P(\theta' = 0.08 \mid \theta = 0.05, \text{}) \approx 0.3 \qquad (\text{，})$$

：$P$ 。 $P$ ； $P$ ""。 RL  $P$ —— $10^{170}$ ，LLM  token 。**" P" RL 。**

###  R

$$R(s, a) = \text{ } s \text{  } a \text{ }$$

 RL ****——，。：

|      |                 |        |
| -------- | ----------------------- | ---------- |
|    |  +1， -1        | 、 |
| CartPole |  +1/              | 、 |
|      |  +1（）/ -1（） | 、 |

Richard Sutton ****（Reward Hypothesis）：""（Sutton & Barto, 2018）[^2]。——、、——，RL 。

 RL 。""：，，。""—— RLHF  DPO 。

###  γ

$$\gamma \in [0, 1]$$

$\gamma$ ""。：， A  1  10 ，； B  9  0 ， 10  20 。？

| γ    |  A  G₀ |  B  G₀      |           |
| ---- | ------------ | ----------------- | ----------------- |
| 0.5  | 10           | 0.5⁹ × 20 = 0.39  | A（） |
| 0.9  | 10           | 0.9⁹ × 20 = 7.7   | A（）     |
| 0.99 | 10           | 0.99⁹ × 20 = 18.3 | B（）     |

γ  1，""，。。

### 

、CartPole  LLM ，：

|  |                | CartPole   | LLM                |
| ---- | -------------------- | ---------- | ---------------------- |
| S    | {}（）     | R⁴（） |  token     |
| A    | { A,  B}         | {, }   |  token   |
| P    | （） |    |      |
| R    | ±1                   | +1/      | /  |
| γ    | —（）            | 0.99       | （） |

：

**S**：——。CartPole  4 。LLM " token"—— token，。

**A**： A  B。CartPole 。LLM " token "—— token 。

**P**：，。CartPole 。LLM ：$P$ —— token， token  softmax 。 LLM ，$P$  $\pi$ 。

**R**： ±1。CartPole  +1。LLM ——，""（RLHF），（DPO）。

**γ**：（）。CartPole  0.99。LLM ——，， $G_t$ 。

，。

##  G_t：

MDP 。，****——"，"。**（episode）**， `reset` ，。CartPole ，，、，， episode。

，。$G_t$ ：，，"，"。

""。 RL ——CartPole  200 ， token。、，**（trajectory）**：

$$s_0, a_0, r_0, \; s_1, a_1, r_1, \; s_2, a_2, r_2, \; \ldots$$

， episode  trajectory； trajectory ""，：，，。

， $(s_0, a_0, r_0)$，。CartPole  4 ？4  +1 ，？：$1 + 1 + 1 + 1 = 4$。 1  10  1 ？

。""， $\gamma$ 。 $G_t$：

$$G_t = r_t + \gamma \, r_{t+1} + \gamma^2 \, r_{t+2} + \gamma^3 \, r_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k \, r_{t+k}$$

 CartPole ： +1，$\gamma = 0.9$， 4 ：

$$G_0 = \underbrace{1}_{r_0} + \underbrace{0.9 \times 1}_{r_1} + \underbrace{0.9^2 \times 1}_{r_2} + \underbrace{0.9^3 \times 1}_{r_3} = 1 + 0.9 + 0.81 + 0.729 = 3.439$$

 +1 ， 3  1  $0.729$——$\gamma = 0.9$ ，" 1 "。

### 

**。** ， CartPole ， +1：

$$G_t = 1 + 1 + 1 + \cdots = \infty$$

 $\infty$，，。，。 $\gamma < 1$  $|R| \leq R_{\max}$ ：

$$G_t \leq R_{\max} \sum_{k=0}^{\infty} \gamma^k = \frac{R_{\max}}{1 - \gamma}$$

| γ     |        |                       |
| ----- | ---------- | ------------------------- |
| 0.9   | 10 R_max   |  ~10        |
| 0.99  | 100 R_max  |  ~100           |
| 0.999 | 1000 R_max |  ~1000  |

**。** ，——""： 100  100 ，。$\gamma$ ""。

### 

$G_t$ ：

$$G_t = r_t + \gamma \, G_{t+1}$$

：$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$， $\gamma$ ：$G_t = r_t + \gamma (r_{t+1} + \gamma r_{t+2} + \cdots) = r_t + \gamma G_{t+1}$。

：** $t$  =  + **。， $G_{t+1}$。

——。，。

##  π：

MDP ，$G_t$ 。——****：。 $\pi$ 。 $G_t$  $\pi^*$。

。

****——：

$$a = \pi(s)$$

Q-learning  DQN —— Q  $\arg\max_a Q(s, a)$。，——，， DQN  $\varepsilon$-。

****——：

$$\pi(a \mid s) = P(a \mid s)$$

——，。

： 1（ 50/50） $\pi(\text{ A}) = 0.5, \pi(\text{ B}) = 0.5$； 2（ A） $\pi(s) = \text{ A}$。

 CartPole ， 4 ，""""。（$\pi(\text{}) \approx 0.5$）， $\pi(\text{}) \approx 0.95$。 1 ""。

 LLM ，—— token （）， token （）。 2  DPO ，。

### 

|        |                      |                    |
| ---------- | ------------------------------ | ---------------------------- |
|        | （ ε-）      |                      |
|      | ， | ， |
|  | ，         | ，       |

2014  Silver  [^3]，。 PPO、A3C ——。

## 

 RL ****——：

**1. MDP  $(S, A, P, R, \gamma)$ 。**  RL ，：（S）、（A）、（P）、（R）、（γ）。、CartPole、LLM ，。

**2.  $G_t$ 。** ，$\gamma < 1$ 。 $G_t = r_t + \gamma G_{t+1}$ 。

**3.  $\pi$ 。**  $a = \pi(s)$ （DQN）， $\pi(a \mid s)$ （PPO）。 $G_t$  $\pi^*$。

****，****。$G_t$ ，""。""，，。，" 80 ， 60 "，。 $V(s)$ ——。

 $V(s)$ 。[V(s) ](./value-bellman)

## 

[^1]: Bellman, R. (1957). _Dynamic Programming_. Princeton University Press.

[^2]: Sutton, R. S., & Barto, A. G. (2018). _Reinforcement Learning: An Introduction_ (2nd ed.). MIT Press.

[^3]: Silver, D., et al. (2014). Deterministic Policy Gradient Algorithms. _ICML 2014_.
