# 6.2 Critic 

 $A(s,a) \approx \delta = r + \gamma V(s') - V(s)$， Critic  $V(s)$ 。 3  [DP、MC、TD](../chapter03_mdp/dp-mc-td)  Critic 。

::: tip 

- [DP/MC/TD ](../chapter03_mdp/dp-mc-td)——
- [](../chapter03_mdp/value-bellman)——DP 
- [TD Error $\delta$](../chapter03_mdp/dp-mc-td)——TD 
  :::

 3 ， $\pi$： $S$  $M$  0.8 、0.2 。：

|  |  |  |  |  |
| -------- | ---- | -------- | -------- | ---- |
| $S$      |  | 0.2      | $S$      | $-2$ |
| $S$      |  | 0.8      | $M$      | $-1$ |
| $M$      |  | 0.2      | $S$      | $-2$ |
| $M$      |  | 0.8      | $G$      | $-1$ |
| $G$      |  | 1.0      |        | $0$  |

 $\gamma=1$。，。

## DP：

 $P$  $R$（：[MDP ](../chapter03_mdp/mdp)），[](../chapter03_mdp/value-bellman) Critic：

$$V_\phi(s) \leftarrow \sum_a \pi(a|s) \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_\phi(s') \right]$$

：

|              |                                             |
| ---------------- | ----------------------------------------------- |
| $V_\phi(s)$      | Critic  $s$ ， $\phi$ |
| $a$              |  $s$ （、）       |
| $\pi(a \mid s)$  |  $s$  $a$           |
| $R(s,a)$         |  $s$  $a$         |
| $s'$             |  $a$                |
| $P(s' \mid s,a)$ |  $s$  $a$  $s'$       |
| $V_\phi(s')$     | Critic  $s'$            |
| $\gamma$         | ，            |

 $S$ 。，。（、）， $\sum_{s'}$  1：

$$
\begin{aligned}
V_\phi(S) &\leftarrow \pi(\text{} \mid S)\left[R(S,\text{}) + V_\phi(M)\right] + \pi(\text{} \mid S)\left[R(S,\text{}) + V_\phi(S)\right] \\
          &= 0.8\left[-1 + V_\phi(M)\right] + 0.2\left[-2 + V_\phi(S)\right]
\end{aligned}
$$

 $M$ ， $G$、 $S$：

$$
\begin{aligned}
V_\phi(M) &\leftarrow \pi(\text{} \mid M)\left[R(M,\text{}) + V_\phi(G)\right] + \pi(\text{} \mid M)\left[R(M,\text{}) + V_\phi(S)\right] \\
          &= 0.8\left[-1 + V_\phi(G)\right] + 0.2\left[-2 + V_\phi(S)\right]
\end{aligned}
$$

，$V_\phi$  $V^\pi$ 。 0 ，。

** 1 **—— 0，：

$$
\begin{aligned}
V_1(S) &= 0.8[-1 + V_0(M)] + 0.2[-2 + V_0(S)] \\
       &= 0.8(-1 + 0) + 0.2(-2 + 0) = -0.8 - 0.4 = -1.2
\end{aligned}
$$

$$
\begin{aligned}
V_1(M) &= 0.8[-1 + V_0(G)] + 0.2[-2 + V_0(S)] \\
       &= 0.8(-1 + 0) + 0.2(-2 + 0) = -0.8 - 0.4 = -1.2
\end{aligned}
$$

** 2 **—— 1 ：

$$
\begin{aligned}
V_2(S) &= 0.8[-1 + V_1(M)] + 0.2[-2 + V_1(S)] \\
       &= 0.8[-1 + (-1.2)] + 0.2[-2 + (-1.2)] \\
       &= 0.8 \times (-2.2) + 0.2 \times (-3.2) = -1.76 - 0.64 = -2.4
\end{aligned}
$$

$$
\begin{aligned}
V_2(M) &= 0.8[-1 + V_1(G)] + 0.2[-2 + V_1(S)] \\
       &= 0.8[-1 + 0] + 0.2[-2 + (-1.2)] \\
       &= 0.8 \times (-1) + 0.2 \times (-3.2) = -0.8 - 0.64 = -1.44
\end{aligned}
$$

** 3 **—— 2 ：

$$
\begin{aligned}
V_3(S) &= 0.8[-1 + V_2(M)] + 0.2[-2 + V_2(S)] \\
       &= 0.8[-1 + (-1.44)] + 0.2[-2 + (-2.4)] \\
       &= 0.8 \times (-2.44) + 0.2 \times (-4.4) = -1.952 - 0.88 = -2.832
\end{aligned}
$$

$$
\begin{aligned}
V_3(M) &= 0.8[-1 + V_2(G)] + 0.2[-2 + V_2(S)] \\
       &= 0.8(-1 + 0) + 0.2[-2 + (-2.4)] \\
       &= -0.8 + 0.2 \times (-4.4) = -0.8 - 0.88 = -1.68
\end{aligned}
$$

：

|  | $V(S)$ | $V(M)$ | $V(G)$ |
| ---- | ------ | ------ | ------ |
| 0    | 0      | 0      | 0      |
| 1    | -1.2   | -1.2   | 0      |
| 2    | -2.4   | -1.44  | 0      |
| 3    | -2.832 | -1.68  | 0      |
|  | -3.375 | -1.875 | 0      |

，$S$  $M$ ""——，，。

，****—— $s$  $Q(s,a)$ （：[](../chapter03_mdp/value-q)）。" →  → "**（Policy Iteration）**，。

， $P$  $R$。DP  Actor-Critic ——" Critic "。

## MC： Critic

 episode，[ $G_t$](../chapter03_mdp/mdp)  Critic。Critic ：

$$L_{\text{Critic}} = \left( G_t - V_\phi(s) \right)^2 \tag{6.3}$$

：

|                 |                                                     |
| ------------------- | ------------------------------------------------------- |
| $L_{\text{Critic}}$ | Critic ，                   |
| $G_t$               |  $t$  episode （MC ） |
| $V_\phi(s)$         | Critic  $s$                         |

$G_t - V_\phi(s)$  Critic —— $G_t$ ， $V_\phi(s)$ 。。

### 

：

$$
S \xrightarrow{-2} S \xrightarrow{-1} M \xrightarrow{-2} S \xrightarrow{-1} M \xrightarrow{-1} G
$$

 $\gamma=1$，， $G_t$：

|  |  |          | $G_t$                      | MC  $G_t$ |
| -------- | ---- | ---------------- | -------------------------------- | ------------- |
|  1   | $S$  | $-2,-1,-2,-1,-1$ | $-2 + (-1) + (-2) + (-1) + (-1)$ | $-7$          |
|  2   | $S$  | $-1,-2,-1,-1$    | $-1 + (-2) + (-1) + (-1)$        | $-5$          |
|  3   | $M$  | $-2,-1,-1$       | $-2 + (-1) + (-1)$               | $-4$          |
|  4   | $S$  | $-1,-1$          | $-1 + (-1)$                      | $-2$          |
|  5   | $M$  | $-1$             | $-1$                             | $-1$          |

### 

 Critic ， $V(S) = 0$、$V(M) = 0$。 1  $S$ ，MC  $G_t = -7$：

$$
L = (G_t - V(S))^2 = (-7 - 0)^2 = 49
$$

（ $\alpha = 0.5$）：

$$
V(S) \leftarrow V(S) - \alpha \cdot \frac{\partial L}{\partial V(S)} = V(S) - \alpha \cdot 2(V(S) - G_t)
$$

 $\frac{\partial L}{\partial V(S)} = 2(V(S) - G_t) = 2(0 - (-7)) = 14$， $\frac{1}{2}$ ，

$$
V(S) \leftarrow V(S) + \alpha (G_t - V(S)) = 0 + 0.5 \times (-7 - 0) = -3.5
$$

：

|  | MC  $G_t$ |     |                                                                |      |
| ------------ | ------------- | ------- | ---------------------------------------------------------------------- | -------- |
|  1  $S$  | $-7$          | 0       | $0 + 0.5 \times (-7 - 0) = -3.5$                                       | $-3.5$   |
|  2  $S$  | $-5$          | $-3.5$  | $-3.5 + 0.5 \times [-5 - (-3.5)] = -3.5 + 0.5 \times (-1.5) = -4.25$   | $-4.25$  |
|  1  $M$  | $-4$          | 0       | $0 + 0.5 \times (-4 - 0) = -2$                                         | $-2$     |
|  3  $S$  | $-2$          | $-4.25$ | $-4.25 + 0.5 \times [-2 - (-4.25)] = -4.25 + 0.5 \times 2.25 = -3.125$ | $-3.125$ |
|  2  $M$  | $-1$          | $-2$    | $-2 + 0.5 \times [-1 - (-2)] = -2 + 0.5 \times 1 = -1.5$               | $-1.5$   |

MC （：[MC ](../chapter03_mdp/dp-mc-td) $V(s) \leftarrow V(s) + \alpha[G_t - V(s)]$）****（），：

1. ** episode ** $G_t$，
2. ****—— episode  $G_t$ 

，MC ： episode， $(s_t, G_t)$ ， Critic  $\phi$。

## TD：

[TD Error](../chapter03_mdp/dp-mc-td)  Critic。Critic ：

$$L_{\text{Critic}} = \left( r + \gamma V_\phi(s') - V_\phi(s) \right)^2 = \delta^2 \tag{6.4}$$

：

|                 |                                              |
| ------------------- | ------------------------------------------------ |
| $L_{\text{Critic}}$ | Critic ， TD Error           |
| $r$                 |                              |
| $\gamma$            |                                          |
| $V_\phi(s')$        | Critic  $s'$             |
| $V_\phi(s)$         | Critic  $s$              |
| $\delta$            | TD Error， $r + \gamma V_\phi(s') - V_\phi(s)$ |

 $\delta^2$  Critic 。$\delta$ ：，" + """。$\delta > 0$ ，$\delta < 0$ 。

### 

 MC ：

$$
S \xrightarrow{-2} S \xrightarrow{-1} M \xrightarrow{-2} S \xrightarrow{-1} M \xrightarrow{-1} G
$$

 0， $\alpha = 0.5$。TD ，****。

** 1 **：$S \xrightarrow{-2} S$。 $V(S) = 0$，$V(S') = V(S) = 0$。

$$
\delta = r + \gamma V(s') - V(s) = -2 + 1 \times 0 - 0 = -2
$$

$$
V(S) \leftarrow V(S) + \alpha \cdot \delta = 0 + 0.5 \times (-2) = -1
$$

** 2 **：$S \xrightarrow{-1} M$。 $V(S) = -1$（），$V(M) = 0$。

$$
\delta = -1 + 1 \times 0 - (-1) = -1 + 0 + 1 = 0
$$

$$
V(S) \leftarrow -1 + 0.5 \times 0 = -1
$$

$\delta = 0$ " $-1$  $V(M) = 0$ " $V(S)$  $-1$，。

** 3 **：$M \xrightarrow{-2} S$。 $V(M) = 0$，$V(S) = -1$。

$$
\delta = -2 + 1 \times (-1) - 0 = -3
$$

$$
V(M) \leftarrow 0 + 0.5 \times (-3) = -1.5
$$

 $V(S) = -1$  1 ——TD 。

** 4 **：$S \xrightarrow{-1} M$。 $V(S) = -1$，$V(M) = -1.5$。

$$
\delta = -1 + 1 \times (-1.5) - (-1) = -1.5
$$

$$
V(S) \leftarrow -1 + 0.5 \times (-1.5) = -1 + (-0.75) = -1.75
$$

** 5 **：$M \xrightarrow{-1} G$。 $V(M) = -1.5$，$V(G) = 0$。

$$
\delta = -1 + 1 \times 0 - (-1.5) = 0.5
$$

$$
V(M) \leftarrow -1.5 + 0.5 \times 0.5 = -1.5 + 0.25 = -1.25
$$

$\delta = 0.5 > 0$， $M$  $V(M)$ ，$V(M)$ 。

### 

|  |          |  |  $V(s)$ | $r$  | $V(s')$ | TD  $r + \gamma V(s')$ | $\delta$ |  $V(s)$ |
| ---- | ---------------------- | ------------ | --------- | ---- | ------- | -------------------------- | -------- | --------- |
| 1    | $S \xrightarrow{-2} S$ | $S$          | 0         | $-2$ | 0       | $-2 + 0 = -2$              | $-2$     | $-1$      |
| 2    | $S \xrightarrow{-1} M$ | $S$          | $-1$      | $-1$ | 0       | $-1 + 0 = -1$              | $0$      | $-1$      |
| 3    | $M \xrightarrow{-2} S$ | $M$          | 0         | $-2$ | $-1$    | $-2 + (-1) = -3$           | $-3$     | $-1.5$    |
| 4    | $S \xrightarrow{-1} M$ | $S$          | $-1$      | $-1$ | $-1.5$  | $-1 + (-1.5) = -2.5$       | $-1.5$   | $-1.75$   |
| 5    | $M \xrightarrow{-1} G$ | $M$          | $-1.5$    | $-1$ | 0       | $-1 + 0 = -1$              | $0.5$    | $-1.25$   |

### TD 

 3 ，$\delta = -3$：

$$
L = \delta^2 = (-3)^2 = 9
$$

：

$$
\frac{\partial L}{\partial V(M)} = -2\delta = -2 \times (-3) = 6
$$

 $-\frac{\partial L}{\partial V(M)}$ ， $V(M)$ 。 $V(M) \leftarrow V(M) + \alpha \cdot \delta$，。

TD （：[TD(0) ](../chapter03_mdp/dp-mc-td) $V(s) \leftarrow V(s) + \alpha[r + \gamma V(s') - V(s)]$）：

1. ** episode **——
2. ****——$V_\phi(s')$ ""
3. ** Actor **——

****：$V_\phi(s')$ ，。[（Bootstrapping）](../chapter03_mdp/dp-mc-td)——。，。

## 

|                         | **DP**   | **MC** | **TD**             |
| ----------------------- | -------- | ------ | ------------------ |
| ** Critic ？**  |  |  | ****       |
| ** episode ？** |    |    |              |
| **？**              |        |      | （） |
| ****                |        |      |                  |
| ****                |        |      |                  |

### MC  TD 

 $S \xrightarrow{-2} S \xrightarrow{-1} M \xrightarrow{-2} S \xrightarrow{-1} M \xrightarrow{-1} G$， 0，$\alpha = 0.5$，$\gamma = 1$。

**MC**——。 1  $S$ ：

$$
G_0 = (-2) + (-1) + (-2) + (-1) + (-1) = -7
$$

$$
V(S) \leftarrow 0 + 0.5 \times (-7 - 0) = -3.5
$$

MC 。

**TD**—— 1 。 1 ：

$$
\delta = -2 + V(S) - V(S) = -2 + 0 - 0 = -2
$$

$$
V(S) \leftarrow 0 + 0.5 \times (-2) = -1
$$

TD  $(-2)$  MC  $(-7)$， TD 。，TD  $V(S)$  $-3.375$。

 $V^\pi$，：MC （$-3.5$），；TD （$-1$），，。

，Actor-Critic  TD  Critic。（[ 7  GAE](../chapter07_ppo/gae-reward-model)），MC  TD —— $\lambda$ ，。

## Critic 

，Actor-Critic ：

1. ****： $s$ ，Actor  $a$， $r$  $s'$
2. ****：Critic  $V_\phi(s)$  $V_\phi(s')$
3. ** TD Error**：$\delta = r + \gamma V_\phi(s') - V_\phi(s)$
4. ** Critic**： $\delta^2$  Critic  $\phi$
5. ** Actor**： $\delta$  Actor  $\theta$

###  walkthrough

 Critic  $V(S) = -1$、$V(M) = -0.5$、$V(G) = 0$，$\gamma = 0.9$，Critic  $\alpha_\phi = 0.1$，Actor  $\alpha_\theta = 0.01$。

** 1 ：**

 $S$，Actor  0.8 、0.2 。， $r = -1$，$s' = M$。

** 2 ：**

$$
V_\phi(S) = -1, \quad V_\phi(M) = -0.5
$$

** 3 ： TD Error**

$$
\delta = r + \gamma V_\phi(s') - V_\phi(s) = -1 + 0.9 \times (-0.5) - (-1) = -1 + (-0.45) + 1 = -0.45
$$

$\delta = -0.45 < 0$， $S$  $M$ —— $-1$  $M$  $-0.45$， $-1.45$， $S$  $-1$。

** 4 ： Critic**

$$
L_{\text{Critic}} = \delta^2 = (-0.45)^2 = 0.2025
$$

（）：

$$
V(S) \leftarrow V(S) + \alpha_\phi \cdot \delta = -1 + 0.1 \times (-0.45) = -1 + (-0.045) = -1.045
$$

Critic  $V(S)$—— $S$ 。

** 5 ： Actor**

$\delta = -0.45$ （）。Actor ：。：

$$
\theta \leftarrow \theta + \alpha_\theta \cdot \delta \cdot \nabla_\theta \log \pi(\text{} \mid S)
$$

$\delta < 0$  $\nabla_\theta \log \pi(\text{} \mid S)$ ， $\pi(\text{} \mid S)$ 。

 $\delta > 0$，，Actor 。

Critic  $\phi$ " $\delta^2$ "——。Actor  $\theta$ " $\delta$ "——。：Critic ，Actor ；Actor ，Critic ，。

## 

[^1]: Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. _Machine Learning_, 3(1), 9-44.

[^2]: Mnih, V., et al. (2016). Asynchronous methods for deep reinforcement learning. _ICML_. [arXiv:1602.01783](https://arxiv.org/abs/1602.01783)
