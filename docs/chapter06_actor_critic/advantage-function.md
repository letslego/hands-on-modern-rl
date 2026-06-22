# 6.1 

 5 ： $V(s)$ ，。，—— Actor  Critic 。

::: tip 

- [REINFORCE ](../chapter05_policy_gradient/reinforce) $\nabla_\theta J \approx \nabla_\theta \log \pi(a|s) \cdot G_t$——
- [ $V(s)$](../chapter03_mdp/value-bellman)——
- [ $Q(s,a)$](../chapter03_mdp/value-q)—— $Q$  $V$ 
- [TD Error](../chapter03_mdp/dp-mc-td) $\delta = r + \gamma V(s') - V(s)$——
  :::

## 

 5  REINFORCE [](../chapter05_policy_gradient/reinforce)：

$$\nabla_\theta J \approx \nabla_\theta \log \pi(a|s) \cdot G_t$$

$G_t$  episode （：[](../chapter03_mdp/mdp)）。 $G_t$ ——、， $G_t$。

 $V(s)$ ：

$$\nabla_\theta J \approx \nabla_\theta \log \pi(a|s) \cdot (G_t - V(s))$$

 $G_t - V(s)$ **（Advantage Function）** 。：

$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s) \tag{6.1}$$

|          |                                                                                                      |
| ------------ | -------------------------------------------------------------------------------------------------------- |
| $A^\pi(s,a)$ | ： $s$  $a$，""。                                                  |
| $Q^\pi(s,a)$ | [](../chapter03_mdp/value-q)： $s$  $a$， $\pi$ 。 |
| $V^\pi(s)$   | [](../chapter03_mdp/value-bellman)： $s$  $\pi$ 。             |
| $\pi$        | ，。                                                                 |

" $a$，"。

：**，""。**

- $A > 0$：，
- $A < 0$：，
- $A \approx 0$：

：$V(s)$ " 60%"，$Q(s, \text{})$ " 75%"。 $A = 75\% - 60\% = 15\%$， 15%，。

 3  episode 。 $\gamma = 0.9$，：

$$s_0 \xrightarrow{r=+2} s_1 \xrightarrow{r=+3} s_2 \xrightarrow{r=+1} s_3\ (\text{})$$

 $G_t$：

$$G_0 = r_1 + \gamma r_2 + \gamma^2 r_3 = 2 + 0.9 \times 3 + 0.9^2 \times 1 = 2 + 2.7 + 0.81 = 5.51$$

$$G_1 = r_2 + \gamma r_3 = 3 + 0.9 \times 1 = 3.9$$

$$G_2 = r_3 = 1$$

 Critic ：

|   | $V(s)$ |
| ----- | ------ |
| $s_0$ | 3.0    |
| $s_1$ | 2.5    |
| $s_2$ | 0.8    |

 $G_t$  $V(s)$  $A \approx G_t - V(s)$，：

|  $t$ |   | $G_t$  | $V(s_t)$ | $A = G_t - V(s_t)$  |               |
| -------- | ----- | ------ | -------- | ------------------- | ----------------- |
| 0        | $s_0$ | $5.51$ | $3.0$    | $5.51 - 3.0 = 2.51$ |  $2.51$ |
| 1        | $s_1$ | $3.9$  | $2.5$    | $3.9 - 2.5 = 1.4$   |  $1.4$  |
| 2        | $s_2$ | $1$    | $0.8$    | $1 - 0.8 = 0.2$     |  $0.2$  |

，。$G_t - V(s)$  MC ；（ $G_t$）。

## 

，**""**，""。

。 $s$ ， $V(s) = 10$。 4 ， $G_t^{(1)} = 18$、$G_t^{(2)} = 15$、$G_t^{(3)} = 7$、$G_t^{(4)} = 4$。

 $G_t$ ：

| Episode | $G_t$ |            |                |
| ------- | ----- | ------------------ | ------------------ |
| 1       | 18    | $\nabla \times 18$ | ， |
| 2       | 15    | $\nabla \times 15$ | ，     |
| 3       | 7     | $\nabla \times 7$  | ，     |
| 4       | 4     | $\nabla \times 4$  | ，     |

。"，"—— episode 3  4 。

 $A = G_t - V(s)$ ：

| Episode | $G_t$ | $V(s)$ | $A = G_t - V(s)$ |              |                  |
| ------- | ----- | ------ | ---------------- | -------------------- | -------------------- |
| 1       | 18    | 10     | $18 - 10 = +8$   | $\nabla \times (+8)$ | ，   |
| 2       | 15    | 10     | $15 - 10 = +5$   | $\nabla \times (+5)$ | ，         |
| 3       | 7     | 10     | $7 - 10 = -3$    | $\nabla \times (-3)$ | ，       |
| 4       | 4     | 10     | $4 - 10 = -6$    | $\nabla \times (-6)$ | ， |

 $G_t$ ， episode ，""""。 $A$ ，：，。

。 $G_t$ ， $\frac{18+15+7+4}{4} = 11$， $\frac{(18-11)^2+(15-11)^2+(7-11)^2+(4-11)^2}{4} = \frac{49+16+16+49}{4} = 32.5$。 $A$ ， $\frac{8+5-3-6}{4} = 1$， $\frac{(8-1)^2+(5-1)^2+(-3-1)^2+(-6-1)^2}{4} = \frac{49+16+16+49}{4} = 32.5$。

， $A$ 。，$G_t$ （ 0 ）， $A$  $V(s)$ ，。""。

##  TD Error 

 $A = Q - V$， $Q$。，。

 $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$ 。[](../chapter03_mdp/value-q)：

$$Q^\pi(s,a) = \mathbb{E}\left[R_{t+1} + \gamma V^\pi(S_{t+1}) \mid S_t = s, A_t = a\right]$$

： $s$  $a$ ，。（ episode，）， $Q$ ：

$$Q(s,a) \approx r + \gamma V(s')$$

 $r$ ，$s'$ 。：

$$A(s,a) = Q(s,a) - V(s) \approx r + \gamma V(s') - V(s)$$

[TD Error](../chapter03_mdp/dp-mc-td)：

$$A(s,a) \approx r + \gamma V(s') - V(s) = \delta \tag{6.2}$$

|      |                                              |
| -------- | ------------------------------------------------ |
| $r$      | 。                       |
| $\gamma$ | ，。             |
| $V(s')$  | Critic  $s'$ 。              |
| $V(s)$   | Critic  $s$ 。               |
| $\delta$ | TD Error：，。 |

 TD Error  $G_t$ ，：

1. ** episode **——（$G_t$ ，[MC ](../chapter03_mdp/dp-mc-td)）
2. ****——$\delta$ （$G_t$ ）

。 $\gamma = 0.9$，：

-  $s$，Critic  $V(s) = 5.0$
- ， $r = +2$
-  $s'$，Critic  $V(s') = 4.0$

 TD Error ：

$$\delta = r + \gamma V(s') - V(s) = 2 + 0.9 \times 4.0 - 5.0 = 2 + 3.6 - 5.0 = +0.6$$

$\delta = +0.6$， Critic  $0.6$。 $\delta$ ，。

。 $r = -1$：

$$\delta = -1 + 0.9 \times 4.0 - 5.0 = -1 + 3.6 - 5.0 = -2.4$$

$\delta = -2.4$，。。

 $\delta = 0$ 。 $r = +1$，$V(s') = 5.0$，$V(s) = 5.5$：

$$\delta = 1 + 0.9 \times 5.0 - 5.5 = 1 + 4.5 - 5.5 = 0$$

$\delta = 0$： Critic ，，。

。 3  episode，$\gamma = 0.9$：

|  |   |   | $r$  |  | $V(s)$ | $V(s')$ | $\delta = r + \gamma V(s') - V(s)$                |
| ---- | ----- | ----- | ---- | -------- | ------ | ------- | ------------------------------------------------- |
| 0    | $s_0$ | $a_0$ | $+3$ | $s_1$    | 2.0    | 4.0     | $3 + 0.9 \times 4.0 - 2.0 = 3 + 3.6 - 2.0 = +4.6$ |
| 1    | $s_1$ | $a_1$ | $+1$ | $s_2$    | 4.0    | 1.0     | $1 + 0.9 \times 1.0 - 4.0 = 1 + 0.9 - 4.0 = -2.1$ |
| 2    | $s_2$ | $a_2$ | $+2$ | $s_3$    | 1.0    | 0.0     | $2 + 0.9 \times 0.0 - 1.0 = 2 + 0.0 - 1.0 = +1.0$ |

 $\delta$  $+4.6$、$-2.1$、$+1.0$。 0 ， $a_0$ ； 1 ， $a_1$ ； 2 ， $a_2$。

 MC  $G_t$ （）：

$$G_0 = 3 + 0.9 \times 1 + 0.9^2 \times 2 = 3 + 0.9 + 1.62 = 5.52$$

$$G_1 = 1 + 0.9 \times 2 = 2.8$$

$$G_2 = 2$$

 MC ：

|  | $G_t$ | $V(s)$ | $A_{\text{MC}} = G_t - V(s)$ |
| ---- | ----- | ------ | ---------------------------- |
| 0    | 5.52  | 2.0    | $5.52 - 2.0 = +3.52$         |
| 1    | 2.8   | 4.0    | $2.8 - 4.0 = -1.2$           |
| 2    | 2     | 1.0    | $2 - 1.0 = +1.0$             |

（、、），。TD  $\delta$ ，MC  $G_t - V(s)$ 。$\delta$ （），（ $V(s')$ ）；$G_t - V(s)$ （）。

 MC → TD ：REINFORCE  $G_t$（MC），Actor-Critic  $\delta$（TD）。

|          | **REINFORCE (MC)**           | **Actor-Critic (TD)**                              |
| -------- | ---------------------------- | -------------------------------------------------- |
|  | $G_t - V(s)$（） | $r + \gamma V(s') - V(s) = \delta$（） |
|  | episode                |                                            |
|      |                            |                                                  |
|      |                            |  Critic                                    |

## Critic 

 $\delta = r + \gamma V(s') - V(s)$， $V(s)$  $V(s')$。，$V$ ——。 **Critic**。

```
Actor（）           Critic（）
  ： s                 ： s
  ：π_θ(a|s)      ：V_φ(s) 
  ：                ：
  ：θ                     ：φ
```

Actor  Critic （ $s$），：Actor ，Critic 。 $A \approx \delta$ ：Critic ，Actor 。

 Critic ？ $V(s)$？ 3  [DP、MC、TD](../chapter03_mdp/dp-mc-td)  Critic 。[Critic ](./critic-training)
