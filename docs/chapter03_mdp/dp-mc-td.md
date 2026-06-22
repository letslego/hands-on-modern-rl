# 3.4 DP、MC、TD：

## 

****

- ：， $V(s)$。
- DP、MC、TD：，“”。
- TD Error：“ + ”。

 $V^\pi(s)$ ： $s$ ， $\pi$ ，。，****。，：

|  | $V(s)$  |
| ---- | ----------------- |
| $S$  | 0                 |
| $M$  | 0                 |
| $G$  | 0                 |

，。“”，：**？**

，。 $s$， $\pi$， $V^\pi(s)$。， $V^\pi(s)$ ：

$$
V^\pi(s)
=
\mathbb{E}_\pi\left[
R_{t+1}+\gamma V^\pi(S_{t+1})
\mid S_t=s
\right].
$$

：，“”。，，，。

， $V^\pi$， $V$。，：，（target）， $V(s)$ 。 target ，。

“” DP、MC、TD。：****， $s$  $a$ ， $s'$，。 $P(s'\mid s,a)$  $R(s,a)$。DP ，；MC ，，；TD ，，“ + ”。，。

> ，？，？

::: info 
DP、MC、TD 。DP ，MC ，TD 。
:::

****

$$
V_{k+1}(s)
\leftarrow
\sum_a\pi(a\mid s)
\left[
R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V_k(s')
\right]
\quad \text{（DP ：）}
$$

$$
V(s)\leftarrow V(s)+\alpha\left[G_t-V(s)\right]
\quad \text{（MC ：）}
$$

$$
V(s)\leftarrow V(s)+\alpha\left[r+\gamma V(s')-V(s)\right]
\quad \text{（TD(0) ：）}
$$

$$
\delta=r+\gamma V(s')-V(s)
\quad \text{（TD Error：）}
$$

> **“”：**
>
> - $V(s)$： $s$ 。
> - $V_k(s)$： $k$ ， $s$ 。
> - $P(s'\mid s,a)$、$R(s,a)$：DP 。
> - $G_t$：MC 。
> - $r+\gamma V(s')$：TD 。
> - $\alpha$：。
> - $\delta$：TD 。

<span id=””></span>

## 

，”，”。： $\pi$ ，，，。， GridWorld ，： $s$ ，？ $V^\pi(s)$。

：

$$
V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s],
\qquad
G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots.
$$

$V^\pi(s)$ ” $s$ 、 $\pi$ ”。。， $s$ 、，。，。

 Sutton  Barto  6  prediction problem： $\pi$， $v_\pi$[^4]。 $v_\pi$，—— $s$ 、 $\pi$ ，。，，。 $V$，。

：**** $v_\pi$， $V(S_t)$。Sutton  Barto  target。

， target。

**DP** 。，，。，，。

**MC** 。 episode ， $G_t$  target。$G_t$ ，。 episode 。

**TD** ， episode 。， $R_{t+1}$  $V(S_{t+1})$  target， $R_{t+1} + \gamma V(S_{t+1})$。

TD [^1]：

$$
G_t = R_{t+1} + \gamma G_{t+1}.
$$

$G_{t+1}$ 。 $t+1$ ，$G_{t+1}$ ，TD  $V(S_{t+1})$ ：

$$
\text{target}_{\mathrm{TD}} = R_{t+1}+\gamma V(S_{t+1}).
$$

（bootstrapping）。 TD ，。

 target ，： target 。

$$
V(s)\leftarrow V(s)+\alpha\left[\text{target}-V(s)\right].
$$

$V(s)$ ，$\text{target}$ ，$\alpha$ 。$\text{target}-V(s)$ ””； $\alpha$，。$\alpha=1$ ，$\alpha$ 。

，， target 。**， target 。**

。，“”：

![](./images/three-state-corridor.svg)

|  |  |  |  |  |
| -------- | ---- | -------- | -------- | ---- |
| $S$      |  | 0.2      | $S$      | $-2$ |
| $S$      |  | 0.8      | $M$      | $-1$ |
| $M$      |  | 0.2      | $S$      | $-2$ |
| $M$      |  | 0.8      | $G$      | $-1$ |
| $G$      |  | 1.0      |        | $0$  |

 $S$ ，$M$ ，$G$ ，。 1 ；，， 2 。$S$ ， $S$；$M$  $S$。

::: details  +1？
：“”，**“”**。，、， $r=-1$， $r=-2$。

 $+1$，****。 $+1$， $+2$，。，，；，，**“”**。

，“”，。，****：$-2$  $-10$ 。，**“”**。，， $0$。

，。“ $+1$， $0$”。。， DP、MC、TD ：****。
:::

，****：** $S$  $M$  $0.8$ 、$0.2$ **。，。，$V(M)$ “”， $M$  $S$；$V(S)$ “”，。

，， $V(S)$  $V(M)$ 。：，DP、MC、TD  $\text{target}$。

## 

：

$$
V^\pi(s) = r_\pi(s) + \gamma \sum_{s'} P_\pi(s' \mid s) V^\pi(s').
$$

 $r_\pi(s)$  $s$ ，$P_\pi(s' \mid s)$ 。（dynamic programming, DP），：****。、 $P(s' \mid s,a)$ 、 $R(s,a)$ 。

 $R(s,a)$  $P(s' \mid s,a)$ 。 $S$ ：

$$
r_\pi(S) = \underbrace{\pi(\text{} \mid S)}_{0.8} \cdot \underbrace{R(S, \text{})}_{-1} + \underbrace{\pi(\text{} \mid S)}_{0.2} \cdot \underbrace{R(S, \text{})}_{-2} = -1.2.
$$

$$
P_\pi(M \mid S) = \underbrace{\pi(\text{} \mid S)}_{0.8} \cdot \underbrace{P(M \mid S, \text{})}_{1} + \underbrace{\pi(\text{} \mid S)}_{0.2} \cdot \underbrace{P(M \mid S, \text{})}_{0} = 0.8,
$$

 $P_\pi(S \mid S) = 0.2$ 。

 $r_\pi(S)$ ，$P_\pi(s' \mid S)$ 。

$$
r_\pi(s) = \sum_a \pi(a \mid s) R(s,a), \quad P_\pi(s' \mid s) = \sum_a \pi(a \mid s) P(s' \mid s,a).
$$

，：

$$
\text{target}_{\mathrm{DP}}(s)=
\sum_a\pi(a\mid s)
\left[
R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V_k(s')
\right].
$$

： $\pi$ ，。：

|                              |                                          |
| -------------------------------- | -------------------------------------------- |
| $\text{target}_{\mathrm{DP}}(s)$ |  $s$ 。          |
| $a$                              |  $s$ ，、。  |
| $\pi(a\mid s)$                   |  $s$  $a$ 。 |
| $R(s,a)$                         |  $a$ ，。      |
| $s'$                             | 。                 |
| $P(s'\mid s,a)$                  |  $s$  $a$ ， $s'$ 。 |
| $V_k(s')$                        |  $k$ ， $s'$ 。  |
| $\gamma$                         | ，。     |

：

$$
R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V_k(s').
$$

：** $a$，，？**  $\sum_{s'}$ 。

：

$$
\sum_a\pi(a\mid s)[\cdots].
$$

：**，。**  DP 。

 $\max_a$。DP ， $\pi$。，。

 target 。：**，**；**， $S$**。：， $0.8$ ， $0.2$ 。DP ，“”，。

 $\gamma=1$。，，、、。， $\sum_{s'}$  1， 0。 **$S$ ** ， $P(M\mid S,\text{})=1$， $V_{\text{old}}(M)$。

 $S$ ，“”“”， $\sum_a$ ：

$$
\begin{aligned}
\text{target}_{\mathrm{DP}}(S)
&=\sum_a\pi(a\mid S)
\left[
R(S,a)+\sum_{s'}P(s'\mid S,a)V_{\text{old}}(s')
\right]\\
&=\pi(\text{}\mid S)
\left[R(S,\text{})+P(M\mid S,\text{})V_{\text{old}}(M)\right]\\
&\quad+\pi(\text{}\mid S)
\left[R(S,\text{})+P(S\mid S,\text{})V_{\text{old}}(S)\right]\\
&=0.8[-1+1\cdot V_{\text{old}}(M)]+0.2[-2+1\cdot V_{\text{old}}(S)].
\end{aligned}
$$

 $M$ ， $G$， $S$：

$$
\begin{aligned}
\text{target}_{\mathrm{DP}}(M)
&=\sum_a\pi(a\mid M)
\left[
R(M,a)+\sum_{s'}P(s'\mid M,a)V_{\text{old}}(s')
\right]\\
&=\pi(\text{}\mid M)
\left[R(M,\text{})+P(G\mid M,\text{})V_{\text{old}}(G)\right]\\
&\quad+\pi(\text{}\mid M)
\left[R(M,\text{})+P(S\mid M,\text{})V_{\text{old}}(S)\right]\\
&=0.8[-1+1\cdot V_{\text{old}}(G)]+0.2[-2+1\cdot V_{\text{old}}(S)].
\end{aligned}
$$

：

|  |                     |                     |   |
| ------------ | --------------------------- | --------------------------- | ----------- |
| $S$          | $0.8[-1+V_{\text{old}}(M)]$ | $0.2[-2+V_{\text{old}}(S)]$ |  $V(S)$ |
| $M$          | $0.8[-1+V_{\text{old}}(G)]$ | $0.2[-2+V_{\text{old}}(S)]$ |  $V(M)$ |

。：

|                                        |                                                                                                             |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| ** $S$ **： $-1$， $M$       | <img src="./images/three-state-corridor-highlight-s-right.svg" alt=" S " style="max-width: 320px;" /> |
| ** $S$ **： $-2$， $S$ | <img src="./images/three-state-corridor-highlight-s-left.svg" alt=" S " style="max-width: 320px;" />  |
| ** $M$ **： $-1$， $G$   | <img src="./images/three-state-corridor-highlight-m-right.svg" alt=" M " style="max-width: 320px;" /> |
| ** $M$ **： $-2$， $S$       | <img src="./images/three-state-corridor-highlight-m-left.svg" alt=" M " style="max-width: 320px;" />  |

，，。

 0 ：

|  | $V(S)$ | $V(M)$ | $V(G)$ |
| -------- | ------ | ------ | ------ |
|  0   | 0      | 0      | 0      |

。 $V_0$  0，：

$$
\begin{aligned}
V_1(S) &= 0.8[-1+V_0(M)] + 0.2[-2+V_0(S)] \\
       &= 0.8(-1+0) + 0.2(-2+0) \\
       &= -1.2.
\end{aligned}
$$

$$
\begin{aligned}
V_1(M) &= 0.8[-1+V_0(G)] + 0.2[-2+V_0(S)] \\
       &= 0.8(-1+0) + 0.2(-2+0) \\
       &= -1.2.
\end{aligned}
$$

：

|  | $V(S)$ | $V(M)$ | $V(G)$ |
| -------------- | ------ | ------ | ------ |
|  1         | -1.2   | -1.2   | 0      |

，：

$$
\begin{aligned}
V_2(S) &= 0.8[-1+V_1(M)] + 0.2[-2+V_1(S)] \\
       &= 0.8[-1+(-1.2)] + 0.2[-2+(-1.2)] \\
       &= -2.4.
\end{aligned}
$$

$$
\begin{aligned}
V_2(M) &= 0.8[-1+V_1(G)] + 0.2[-2+V_1(S)] \\
       &= 0.8(-1+0) + 0.2[-2+(-1.2)] \\
       &= -1.44.
\end{aligned}
$$

：

|  | $V(S)$ | $V(M)$ | $V(G)$ |
| -------------- | ------ | ------ | ------ |
|  2         | -2.4   | -1.44  | 0      |

$$
\begin{aligned}
V_3(S) &= 0.8[-1+V_2(M)] + 0.2[-2+V_2(S)] \\
       &= 0.8[-1+(-1.44)] + 0.2[-2+(-2.4)] \\
       &= -2.832.
\end{aligned}
$$

$$
\begin{aligned}
V_3(M) &= 0.8[-1+V_2(G)] + 0.2[-2+V_2(S)] \\
       &= 0.8(-1+0) + 0.2[-2+(-2.4)] \\
       &= -1.68.
\end{aligned}
$$

，：

|  | $V(S)$ | $V(M)$ | $V(G)$ |
| ---- | ------ | ------ | ------ |
| 0    | 0      | 0      | 0      |
| 1    | -1.2   | -1.2   | 0      |
| 2    | -2.4   | -1.44  | 0      |
| 3    | -2.832 | -1.68  | 0      |
|  | -3.375 | -1.875 | 0      |

 DP ：，“”。，；。，$S$  $M$ ，，。

，DP 。： $s$  $a$， $\pi$ ，？， $V^\pi$ ：

$$
Q^\pi(s,a)=R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^\pi(s'),
$$

：

$$
\pi'(s)=\arg\max_a Q^\pi(s,a).
$$

：，。，，DP ： $P$  $R$ 。、。，DP ：，。

## 

（Monte Carlo, MC）：**，**。

Monte Carlo ，**、**。 1940 ，·；，。，，。

DP  $P(s'\mid s,a)$  $R(s,a)$，。MC ：****。

。，，。， $t$ 

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

。MC  target：

$$
\text{target}_{\mathrm{MC}} = G_t.
$$

， $V^\pi(s)$。，，：

$$
V(s)\leftarrow V(s)+\alpha\left[G_t-V(s)\right].
$$

 $G_t-V(s)$ “”“”。，；，。

， $S$ 。：

$$
S\xrightarrow{-1}M\xrightarrow{-1}G.
$$

， $M$  $S$。：

$$
S\xrightarrow{-2}S\xrightarrow{-1}M\xrightarrow{-2}S\xrightarrow{-1}M\xrightarrow{-1}G.
$$

：

|  |            |                                                                                                                         |
| ---- | ------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| 1    | **$S\xrightarrow{-2}S$** | <img src="./images/three-state-corridor-highlight-s-left.svg" alt=" 1 ：S " style="max-width: 260px;" />  |
| 2    | **$S\xrightarrow{-1}M$** | <img src="./images/three-state-corridor-highlight-s-right.svg" alt=" 2 ：S  M" style="max-width: 260px;" /> |
| 3    | **$M\xrightarrow{-2}S$** | <img src="./images/three-state-corridor-highlight-m-left.svg" alt=" 3 ：M  S" style="max-width: 260px;" />  |
| 4    | **$S\xrightarrow{-1}M$** | <img src="./images/three-state-corridor-highlight-s-right.svg" alt=" 4 ：S  M" style="max-width: 260px;" /> |
| 5    | **$M\xrightarrow{-1}G$** | <img src="./images/three-state-corridor-highlight-m-right.svg" alt=" 5 ：M  G" style="max-width: 260px;" /> |

MC 。，。 $\gamma=1$，：

|  |  |  | MC  $G_t$ |
| -------- | ---- | ------------------ | ------------- |
|  1   | $S$  | $-2,-1,-2,-1,-1$   | $-7$          |
|  2   | $S$  | $-1,-2,-1,-1$      | $-5$          |
|  3   | $M$  | $-2,-1,-1$         | $-4$          |
|  4   | $S$  | $-1,-1$            | $-2$          |
|  5   | $M$  | $-1$               | $-1$          |

。：

$$
V(s)\leftarrow V(s)+\alpha\left[G_t-V(s)\right].
$$

 0， $\alpha=0.5$， MC。

$$
V(s) \leftarrow V(s) + 0.5\left[G_t - V(s)\right].
$$

$G_t$ 。 1  $S$ ， $-2,-1,-2,-1,-1$， $G_t=-7$； 2  $S$ ， $-1,-2,-1,-1$， $G_t=-5$。

|  | MC  |   |                            |
| ------------ | ------- | ----- | ------------------------------ |
|  1  $S$  | $-7$    | 0     | $0+0.5(-7-0)=-3.5$             |
|  2  $S$  | $-5$    | -3.5  | $-3.5+0.5[-5-(-3.5)]=-4.25$    |
|  1  $M$  | $-4$    | 0     | $0+0.5(-4-0)=-2$               |
|  3  $S$  | $-2$    | -4.25 | $-4.25+0.5[-2-(-4.25)]=-3.125$ |
|  2  $M$  | $-1$    | -2    | $-2+0.5[-1-(-2)]=-1.5$         |

MC  $G_t$ 。episode ，， $G_t$，。 episode ，，， $G_t$，。

MC ，，。 MC 。

 $G_t$ 。，，。，，。

## 

（temporal difference, TD）：**，，**[^3]。

Temporal difference ""——。 DP ， MC 。

 MC  TD，：** $G_t$**。MC  episode ；TD 。

TD 。：

$$
V^\pi(s) = \mathbb{E}_\pi\left[R_{t+1} + \gamma V^\pi(S_{t+1}) \mid S_t = s\right].
$$

” + ”。：，$R_{t+1}$  $S_{t+1}$ ， $V^\pi(S_{t+1})$  $V(S_{t+1})$， TD ：

$$
\text{target}_{\mathrm{TD}} = R_{t+1} + \gamma V(S_{t+1}).
$$



$$
V(s)\leftarrow V(s)+\alpha\left[r+\gamma V(s')-V(s)\right].
$$

 TD （TD error）：

$$
\delta=r+\gamma V(s')-V(s).
$$

。$\delta>0$ ，；$\delta<0$ ，；$\delta=0$ 。

：

$$
S\xrightarrow{-2}S\xrightarrow{-1}M\xrightarrow{-2}S\xrightarrow{-1}M\xrightarrow{-1}G.
$$

 0， $\alpha=0.5$。，。：

$$
\text{}=\text{}+0.5(\text{TD }-\text{}).
$$

 TD ****。 1  $S$ ， 3  $M$  $S$， $-1$  $V(S)$。 TD  MC ：MC ；TD 。

|  |                                                                                                                    |            |  | TD  $r+V(s')$     |                          |
| ---- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------ | ------------ | --------------------- | -------------------------------- |
| 1    | <img src="./images/three-state-corridor-highlight-s-left.svg" alt="TD  1 ：S " style="max-width: 220px;" />  | **$S\xrightarrow{-2}S$** | $V(S)=0$     | $-2+V(S)=-2+0=-2$     | $V(S)=0+0.5(-2-0)=-1$            |
| 2    | <img src="./images/three-state-corridor-highlight-s-right.svg" alt="TD  2 ：S  M" style="max-width: 220px;" /> | **$S\xrightarrow{-1}M$** | $V(S)=-1$    | $-1+V(M)=-1+0=-1$     | $V(S)=-1+0.5[-1-(-1)]=-1$        |
| 3    | <img src="./images/three-state-corridor-highlight-m-left.svg" alt="TD  3 ：M  S" style="max-width: 220px;" />  | **$M\xrightarrow{-2}S$** | $V(M)=0$     | $-2+V(S)=-2-1=-3$     | $V(M)=0+0.5(-3-0)=-1.5$          |
| 4    | <img src="./images/three-state-corridor-highlight-s-right.svg" alt="TD  4 ：S  M" style="max-width: 220px;" /> | **$S\xrightarrow{-1}M$** | $V(S)=-1$    | $-1+V(M)=-1-1.5=-2.5$ | $V(S)=-1+0.5[-2.5-(-1)]=-1.75$   |
| 5    | <img src="./images/three-state-corridor-highlight-m-right.svg" alt="TD  5 ：M  G" style="max-width: 220px;" /> | **$M\xrightarrow{-1}G$** | $V(M)=-1.5$  | $-1+V(G)=-1+0=-1$     | $V(M)=-1.5+0.5[-1-(-1.5)]=-1.25$ |

， TD ：

|  |          | $V(S)$ | $V(M)$ |                                                    |
| ---- | -------------------- | ------ | ------ | ------------------------------------------------------ |
|  |              | 0      | 0      | 。                                             |
| 1    | $S\xrightarrow{-2}S$ | -1     | 0      | $S$ ， $-2$  $V(S)$ 。           |
| 2    | $S\xrightarrow{-1}M$ | -1     | 0      |  $M$， $V(M)=0$，$V(S)$ 。                 |
| 3    | $M\xrightarrow{-2}S$ | -1     | -1.5   | $M$  $S$，$V(S)$  $-1$， $V(M)$ 。 |
| 4    | $S\xrightarrow{-1}M$ | -1.75  | -1.5   | $S$  $V(M)=-1.5$，$V(S)$ 。              |
| 5    | $M\xrightarrow{-1}G$ | -1.75  | -1.25  | $M$  $G$， $-1$，$V(M)$ 。           |

MC  TD 。MC  episode ；TD 。

 1  $S\xrightarrow{-2}S$ 。MC  episode  $-7$（）；TD  $-2+V(S)=-2$（）。TD ，。，。

（bootstrapping）。 TD 、，： $V(s')$ ，$r+\gamma V(s')$ 。TD ：，。

TD 。Critic 、、GAE ， TD 。：TD 。

## 

DP、MC  TD ，。

——（3.3 ）：

$$
V^\pi(s) = \mathbb{E}_\pi\left[R_{t+1} + \gamma V^\pi(S_{t+1}) \mid S_t = s\right].
$$

 $V(s)$，。， $V$， target 。

，""。， target。

**DP：，。**

 $P(s'\mid s,a)$  $R(s,a)$ ，。，：

$$
\text{target}_{\mathrm{DP}}(s)
=
\sum_a
\underbrace{\pi(a\mid s)}_{\text{：}}
\left[
\underbrace{R(s,a)}_{\text{}}
+\gamma
\sum_{s'}
\underbrace{P(s'\mid s,a)}_{\text{}}
\underbrace{V_k(s')}_{\text{}}
\right].
$$

DP ，。

**MC：，。**

$P$  $R$ ，。 $V^\pi(s)$ ：

$$
V^\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s].
$$

 episode ， $t$ 

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots
$$

。MC  $G_t$  target：

$$
\text{target}_{\mathrm{MC}} = G_t.
$$

， MC ， episode 。

**TD：，。**

MC ， $G_t$。

$$
G_t = r_{t+1} + \gamma G_{t+1},
$$

 $G_{t+1}$  $V_k(S_{t+1})$，：

$$
\text{target}_{\mathrm{TD}} = r_{t+1} + \gamma V_k(S_{t+1}).
$$

TD ，，。

 target ，：

|               |  target                                                |            |                            |                           |
| ----------------- | ---------------------------------------------------------------- | ------------------ | ---------------------------------- | ----------------------------- |
|  target | $\mathbb{E}_\pi[r+\gamma V(s')\mid s]$                           |                  | 、、 |               |
| DP                | $\sum_a\pi(a\mid s)[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V_k(s')]$ |            |  $R,P$               | $\text{target}_{\mathrm{DP}}$ |
| MC                | $\mathbb{E}_\pi[G_t\mid S_t=s]$                                  |  $R,P$       |      | $G_t$                         |
| TD                | $G_t=r_{t+1}+\gamma G_{t+1}$                                     |  $G_{t+1}$ |  $V_k(S_{t+1})$  | $r_{t+1}+\gamma V_k(S_{t+1})$ |

：

$$
\underbrace{\mathbb{E}_\pi[r + \gamma V(s') \mid s]}_{\text{}}
\xrightarrow{\text{}}
\underbrace{\text{target}_{\mathrm{DP}}}_{\text{}}
\xrightarrow{\text{}}
\underbrace{G_t}_{\text{MC：}}
\xrightarrow{\text{}}
\underbrace{r + \gamma V_k(s')}_{\text{TD：}}
$$

，。 DP ； MC ；， TD 。

，：

$$
V(S)=-3.375,\qquad V(M)=-1.875,\qquad V(G)=0.
$$

，。DP ，；MC ；TD 。

。：$S$  $M$  0.8 、0.2 。DP  1000 ；MC  TD  100  episode， 5 。 $V^\pi$。

```python
import random

STATES = ["S", "M", "G"]
GAMMA = 1.0


def step(state, action):
    # ：，、。
    #  sample_action()。
    if state == "S":
        return ("M", -1) if action == "right" else ("S", -2)
    if state == "M":
        return ("G", -1) if action == "right" else ("S", -2)
    return "G", 0


def sample_action():
    #  pi：80% ，20% 。
    return "right" if random.random() < 0.8 else "left"


def dp_policy_evaluation(n_iter=1_000):
    V = {s: 0.0 for s in STATES}
    for _ in range(n_iter):
        # DP ，。
        #  old ：，。
        old = V.copy()
        V["S"] = 0.8 * (-1 + GAMMA * old["M"]) + 0.2 * (-2 + GAMMA * old["S"])
        V["M"] = 0.8 * (-1 + GAMMA * old["G"]) + 0.2 * (-2 + GAMMA * old["S"])
        V["G"] = 0.0
    return V


def generate_episode():
    # MC  TD ，。
    episode = []
    state = "S"
    while state != "G":
        action = sample_action()
        next_state, reward = step(state, action)
        episode.append((state, reward, next_state))
        state = next_state
    return episode


def mc_every_visit(n_episodes=1_000_000, seed=0):
    random.seed(seed)
    V = {s: 0.0 for s in STATES}
    N = {s: 0 for s in STATES}
    for _ in range(n_episodes):
        episode = generate_episode()
        G = 0.0
        # MC ， G_t。
        for state, reward, _ in reversed(episode):
            G = reward + GAMMA * G
            N[state] += 1
            # ；1/N 。
            V[state] += (G - V[state]) / N[state]
    return V


def td_zero(n_episodes=1_000_000, seed=0):
    random.seed(seed)
    V = {s: 0.0 for s in STATES}
    N = {s: 0 for s in STATES}
    for _ in range(n_episodes):
        state = "S"
        while state != "G":
            action = sample_action()
            next_state, reward = step(state, action)
            N[state] += 1
            alpha = 1.0 / N[state]
            # TD ： + 。
            target = reward + GAMMA * V[next_state]
            V[state] += alpha * (target - V[state])
            state = next_state
    return V


def show(name, values):
    print(f"{name}: S={values['S']:.6f}, M={values['M']:.6f}, G={values['G']:.6f}")


def summarize(name, runs):
    mean_s = sum(v["S"] for v in runs) / len(runs)
    mean_m = sum(v["M"] for v in runs) / len(runs)
    min_s, max_s = min(v["S"] for v in runs), max(v["S"] for v in runs)
    min_m, max_m = min(v["M"] for v in runs), max(v["M"] for v in runs)
    print(
        f"{name}: mean S={mean_s:.6f} [{min_s:.6f}, {max_s:.6f}], "
        f"mean M={mean_m:.6f} [{min_m:.6f}, {max_m:.6f}]"
    )


print("single run")
show("DP", dp_policy_evaluation())
show("MC", mc_every_visit(seed=0))
show("TD", td_zero(seed=0))

print("\n5-run summary")
seeds = range(5)
summarize("MC", [mc_every_visit(seed=s) for s in seeds])
summarize("TD", [td_zero(seed=s) for s in seeds])
```

。DP ；MC  TD  DP：

```text
single run
DP: S=-3.375000, M=-1.875000, G=0.000000
MC: S=-3.373813, M=-1.874359, G=0.000000
TD: S=-3.374871, M=-1.874966, G=0.000000
```

 5 ，：

```text
5-run summary
MC: mean S=-3.374261 [-3.376874, -3.372061], mean M=-1.874122 [-1.874833, -1.872401]
TD: mean S=-3.375231 [-3.380956, -3.366307], mean M=-1.874380 [-1.876858, -1.870551]
```

DP ；MC  TD ，。，。

。DQN  Actor-Critic  TD ，，。REINFORCE ， MC 。DP、MC、TD ”、、”。

## 

。

1.  $V^\pi(s)$  $s$  $\pi$ 。
2. DP ，。，。
3. MC ， $G_t$ 。， episode ，。
4. TD ，。 $r+\gamma V(s')$ ，、，，。
5. DP、MC、TD ：。、，。

 $V(s)$  $Q(s,a)$。$V(s)$ ， $Q(s,a)$ “ $s$  $a$”，。

：[](./value-q)

## 

[^1]: Bellman, R. (1957). _Dynamic Programming_. Princeton University Press.

[^2]: Metropolis, N., & Ulam, S. (1949). The Monte Carlo method. _Journal of the American Statistical Association_, 44(247), 335-341.

[^3]: Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. _Machine Learning_, 3(1), 9-44.

[^4]: Sutton, R. S., & Barto, A. G. (2018). _Reinforcement Learning: An Introduction_ (2nd ed.). MIT Press, Chapter 6, Section 6.1 and Chapter 7, Section 7.1. Section 6.1 explicitly compares the Monte Carlo target, TD target, and DP target; Section 7.1 explicitly calls the return used in a backup the target of the backup.
