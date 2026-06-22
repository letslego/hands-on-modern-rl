# 5.1 

## 

****

-  4  DQN ： $Q(s,a)$， $\arg\max$ 。
-  Value-Based ：。
-  Policy-Based  $\pi_\theta(a|s)$，、。

## DQN 

 4  DQN ： $Q(s,a)$，， $\arg\max_a Q(s,a)$ 。：""，""，。—— $Q$  $\arg\max$ 。

 CartPole 。 $s = [0.05,\; 0.1,\; -0.02,\; 0.3]$（、、、），DQN ， $Q$ ：

|    | $Q(s,a)$ |
| ------ | -------- |
|  | $0.8$    |
|  | $1.2$    |

$\arg\max$ ：

$$
a^* = \arg\max_a Q(s,a) = \arg\max\{0.8,\; 1.2\} = \text{}.
$$

：， $Q$ 。CartPole  2 ， 2 ；LunarLander  4 ， 4 。 10 、100 、 1000 ，$\arg\max$ —— $Q$ ，，，。

|  |  $Q$  | $\arg\max$  |    |
| -------- | ----------------- | ------------------- | -------- |
| 2        | 2                 | 1                   |      |
| 4        | 4                 | 3                   |      |
| 1000     | 1000              | 999                 |      |
| $10^6$   | $10^6$            | $10^6 - 1$          |  |
| $\infty$ | $\infty$          | $\infty$            |    |

。，， $Q$ ，。

## $\arg\max$ 

$\arg\max$  $Q$ 。，。，。

### ：

。、、， $\tau \in [-10, 10]$。 6 ， $[-10, 10]^6$——。 $Q$ ， $\arg\max$。

：， $\arg\max$。 100 ，。6  100 ，：

$$
N = 100^6 = 10^{12}.
$$

$10^{12}$ ， $Q$ 。 $1\mu\text{s}$（$10^{-6}$ ），****：

$$
T = 10^{12} \times 10^{-6}\text{s} = 10^6\text{s} \approx 11.6 \text{ }.
$$

11.6 。，——，， $1\text{ms}$：

|          |                        |                  |
| ------------ | ------------------------------------------ | -------------------- |
| DQN +  | $10^{12}$                        | $\approx 11.6$     |
|      | 1 ， $\mu = f_\theta(s)$ | $\approx 1\text{ms}$ |

 6 、 100 。、，。****：，。

### ：

。 token 。 50,000，$\arg\max$ —— 50,000 ，。，。

， token ：

| token | $P(\text{token} \mid \text{})$ |
| ----- | ------------------------------------ |
| ""  | $0.40$                               |
| ""  | $0.25$                               |
| ""  | $0.15$                               |
| ""  | $0.10$                               |
| ...   | ...                                  |

$\arg\max$（） ""。，。， 25%  ""，15%  ""——。 $\pi_\theta(a|s)$，。

## 

""，：** $Q$ ， $\pi_\theta(a|s)$**。""，""。

 3 [： $J(\theta)$](../chapter03_mdp/policy-objective)—— $J(\theta)$， $\theta$  $J(\theta)$ 。

：Value-Based ，，；Policy-Based ，，、。

### 

 $\pi_\theta(a|s)$ ，。 CartPole ： $s = [0.05, 0.1, -0.02, 0.3]$，， Softmax ：

$$
\pi_\theta(\text{} \mid s) = 0.3, \quad \pi_\theta(\text{} \mid s) = 0.7.
$$

|                           |                                    |
| ----------------------------- | -------------------------------------- |
| $\pi_\theta$                  |  $\theta$              |
| $\pi_\theta(a \mid s)$        |  $s$  $a$        |
| $s = [0.05, 0.1, -0.02, 0.3]$ | （、、、） |
| $[0.3, 0.7]$                  |                  |

****。 $[0.3, 0.7]$ ： $[0,1)$  $u$， $u < 0.3$ ，。 $u = 0.65$， $0.65 > 0.3$，""。

###  DQN 

，：

**DQN ：**  $Q$  $\to$ $\arg\max$ $\to$ 

$$
[0.8,\; 1.2] \;\xrightarrow{\arg\max}\; \text{} \quad (\text{}).
$$

**：**  $\to$  $\to$ 

$$
[0.3,\; 0.7] \;\xrightarrow{\text{}}\; \begin{cases} \text{} & \text{ } 0.3 \\ \text{} & \text{ } 0.7 \end{cases}
$$

：DQN （）；（）。，——， $\varepsilon$-greedy。

，。 6 ， $\mu_\theta(s) \in \mathbb{R}^6$  $\sigma_\theta(s) \in \mathbb{R}^6$， $\mathcal{N}(\mu_\theta(s),\; \text{diag}(\sigma_\theta^2(s)))$ 。， $\arg\max$，。

## 

|            | Value-Based（DQN）                   | Policy-Based（）                 |
| ---------- | ------------------------------------ | ---------------------------------------- |
|      | $Q(s,a)$：           | $\pi_\theta(a\|s)$： |
|  | $\arg\max_a Q(s,a)$（）      |  $\pi_\theta(\cdot\|s)$          |
|    | （）               | （）                   |
|    |                                |  +                               |
|    | （$\varepsilon$-greedy）     | （）             |
|    | Off-policy（） | On-policy（）      |
|        | （TD ）                | （）                 |
|    | DQN（ 4 ）                       | REINFORCE（） → PPO（ 7 ）       |

。

****——。DQN  $\arg\max$ 。—— Softmax，，。

****——DQN （ $\arg\max$）， $\varepsilon$-greedy（：[DQN ](../chapter04_dqn/dqn-components)）。$\varepsilon$ ，，。，—— 30% ， 30% 。

****——。DQN  off-policy ：，。 on-policy ： $\mathbb{E}_{\pi_\theta}$ 。，。 DQN，。

### 

。：3  $\{s_1, s_2, s_3\}$，2  $\{a_1, a_2\}$， $\gamma = 0.9$。

**DQN ： $Q$ ，$\arg\max$ **

，DQN  $Q$ ：

|   | $Q(s, a_1)$ | $Q(s, a_2)$ |
| ----- | ----------- | ----------- |
| $s_1$ | $1.5$       | $2.3$       |
| $s_2$ | $0.8$       | $-0.4$      |
| $s_3$ | $3.1$       | $2.9$       |

 $\arg\max$：

$$
\pi(s_1) = \arg\max\{1.5,\; 2.3\} = a_2,
$$

$$
\pi(s_2) = \arg\max\{0.8,\; -0.4\} = a_1,
$$

$$
\pi(s_3) = \arg\max\{3.1,\; 2.9\} = a_1.
$$

：。， $\varepsilon$-greedy， $\varepsilon = 0.1$  10% ：

|   |  $a_1$                |  $a_2$                |
| ----- | ----------------------------- | ----------------------------- |
| $s_1$ | $0.1 \times 0.5 = 0.05$       | $0.9 + 0.1 \times 0.5 = 0.95$ |
| $s_2$ | $0.9 + 0.1 \times 0.5 = 0.95$ | $0.1 \times 0.5 = 0.05$       |
| $s_3$ | $0.9 + 0.1 \times 0.5 = 0.95$ | $0.1 \times 0.5 = 0.05$       |

$\varepsilon$-greedy ：10%  $a_1$  $a_2$ 。 $Q(s_3, a_2) = 2.9$  $Q(s_3, a_1) = 3.1$ （）， $s_1$ 。

**： $\pi_\theta(a|s)$，**

：

|   | $\pi(a_1 \mid s)$ | $\pi(a_2 \mid s)$ |
| ----- | ----------------- | ----------------- |
| $s_1$ | $0.2$             | $0.8$             |
| $s_2$ | $0.9$             | $0.1$             |
| $s_3$ | $0.55$            | $0.45$            |

 $s_3$ ，（$0.55$ vs $0.45$），； $s_1$  $s_2$ ，，。 $\varepsilon$，""。

：

|          | DQN  $s_3$                             |  $s_3$                                       |
| ---------------- | ---------------------------------------- | ------------------------------------------------------ |
|          | $Q(s_3, a_1) = 3.1$, $Q(s_3, a_2) = 2.9$ | $\pi(a_1 \mid s_3) = 0.55$, $\pi(a_2 \mid s_3) = 0.45$ |
|        | $\arg\max\{3.1, 2.9\} = a_1$             | ：55%  $a_1$，45%  $a_2$                   |
|              |  $\varepsilon$-greedy（）    | （）                                 |
|  | （ $Q$ ）      | （）                               |

：DQN  $\arg\max$ ；""，""，。

## 

，。 6  Actor-Critic ：，。， Policy-Based 。

，， REINFORCE ：[REINFORCE ](./reinforce)。
