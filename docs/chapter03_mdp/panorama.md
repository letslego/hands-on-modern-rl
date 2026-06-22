# 3.9 ：MDP、

## 

、、、、、Q-Learning、。， 3.1  3.8 ，。

：

1.  MDP 。
2. ，。
3. 。
4. 。
5. DP、MC、TD 。
6. 。
7.  On-policy/Off-policy  Online/Offline。
8. ，。

 Q 、、Actor-Critic、PPO 。

## 

 3.1  3.8 。、。

### 3.1 

$$
\mathbb{E}[R_a] = p_a \cdot (+1) + (1-p_a)\cdot(-1) = 2p_a - 1
\quad \text{（；：； 3.1）}
$$

$$
\mathbb{E}[R_T] = \mathbb{E}[R_{a_1}] + \mathbb{E}[R_{a_2}] + \cdots + \mathbb{E}[R_{a_T}] = \sum_{t=1}^{T} \mathbb{E}[R_{a_t}]
\quad \text{（T ；：； 3.1）}
$$

$$
\mathrm{Regret}(T) = T\mu^* - \sum_{t=1}^{T}\mu_{a_t}, \qquad \mu^*=\max_a \mu_a
\quad \text{（Regret；：； 3.1）}
$$

### 3.2 MDP

$$
\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle
\quad \text{（MDP ；：； 3.2）}
$$

$$
P(s' \mid s,a), \qquad R(s,a), \qquad \gamma \in [0,1]
\quad \text{（、；：、； 3.2）}
$$

$$
G_t = \sum_{k=0}^{\infty}\gamma^k r_{t+k} = r_t + \gamma G_{t+1}
\quad \text{（；： t ； 3.2）}
$$

$$
a = \pi(s), \qquad \pi(a\mid s)=P(a\mid s)
\quad \text{（；：； 3.2）}
$$

### 3.3 V(s) 

$$
V^\pi(s)=\mathbb{E}_\pi\left[\sum_{k=0}^{\infty}\gamma^k r_{t+k}\mid s_t=s\right]
\quad \text{（；：； 3.3）}
$$

$$
V^\pi(s)=\sum_{a\in\mathcal{A}}\pi(a\mid s)\left[R(s,a)+\gamma\sum_{s'\in\mathcal{S}}P(s'\mid s,a)V^\pi(s')\right]
\quad \text{（；：； 3.3）}
$$

$$
V^*(s)=\max_a\left[R(s,a)+\gamma\sum_{s'\in\mathcal{S}}P(s'\mid s,a)V^*(s')\right]
\quad \text{（；：； 3.3）}
$$

$$
\text{Target}=r+\gamma V(s'), \qquad \delta=\text{Target}-V(s)
\quad \text{（Bellman Target  TD Error ；：； 3.3）}
$$

### 3.4 DP、MC、TD

$$
V(s) \leftarrow \sum_a \pi(a\mid s)\left[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V(s')\right]
\quad \text{（DP ；：； 3.4）}
$$

$$
\pi'(s)=\arg\max_a\left[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^\pi(s')\right]
\quad \text{（；：； 3.4）}
$$

$$
V(s) \leftarrow V(s)+\alpha\left[G_t-V(s)\right]
\quad \text{（MC ；：； 3.4）}
$$

$$
V(s) \leftarrow V(s)+\alpha\left[r+\gamma V(s')-V(s)\right]
\quad \text{（TD(0) ；：； 3.4）}
$$

$$
\delta = r+\gamma V(s')-V(s)
\quad \text{（TD Error；：； 3.4）}
$$

### 3.5 Q(s,a)

$$
Q^\pi(s,a)=\mathbb{E}_\pi\left[G_t\mid s_t=s,a_t=a\right]
\quad \text{（；： s  a ； 3.5）}
$$

$$
V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a)
\quad \text{（V-Q ；：； 3.5）}
$$

$$
Q^\pi(s,a)=R(s,a)+\gamma\sum_{s'\in\mathcal{S}}P(s'\mid s,a)\sum_{a'\in\mathcal{A}}\pi(a'\mid s')Q^\pi(s',a')
\quad \text{（Q ；：； 3.5）}
$$

$$
Q^*(s,a)=R(s,a)+\gamma\sum_{s'\in\mathcal{S}}P(s'\mid s,a)\max_{a'}Q^*(s',a')
\quad \text{（Q ；：； 3.5）}
$$

$$
\pi^*(s)=\arg\max_a Q^*(s,a)
\quad \text{（；：； 3.5）}
$$

### 3.5 Q-Learning

$$
\text{TD Target}=r+\gamma\max_{a'}Q(s',a')
\quad \text{（Q-Learning TD ；：； 3.5）}
$$

$$
\delta=r+\gamma\max_{a'}Q(s',a')-Q(s,a)
\quad \text{（Q-Learning TD Error；： TD ； 3.5）}
$$

$$
Q(s,a)\leftarrow Q(s,a)+\alpha\left[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\right]
\quad \text{（Q-Learning ；：-； 3.5）}
$$

$$
a_t=
\begin{cases}
\text{} & \text{ }\varepsilon\\
\arg\max_a Q(s_t,a) & \text{ }1-\varepsilon
\end{cases}
\quad \text{（}\varepsilon\text{-；：； 3.5）}
$$

### 3.6 

$$
\pi_\theta(a\mid s)=P_\theta(a\mid s)
\quad \text{（；： theta ； 3.6）}
$$

$$
J(\theta)=\mathbb{E}_{\pi_\theta}\left[G_t\right]
=\mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{\infty}\gamma^t r_t\right]
\quad \text{（；：； 3.6）}
$$

$$
\theta^*=\arg\max_\theta J(\theta)
\quad \text{（；：； 3.6）}
$$

$$
\nabla_\theta J(\theta)\propto
\mathbb{E}_{\pi_\theta}\left[\nabla_\theta\log\pi_\theta(a\mid s)\cdot G_t\right]
\quad \text{（；：； 3.6）}
$$

### 3.8 

$$
R(s,a)=
\begin{cases}
+1 & \text{}\\
0 & \text{}\\
-1 & \text{}
\end{cases}
\quad \text{（；：； 3.8）}
$$

$$
R_{\text{shaping}}(s,a,s')=-\left(\text{dist}(s',\text{goal})-\text{dist}(s,\text{goal})\right)
\quad \text{（；：； 3.8）}
$$

$$
F(s,a,s')=\gamma\Phi(s')-\Phi(s)
\quad \text{（；：； 3.8）}
$$

$$
r_t^{\text{intrinsic}}=\left\|f(s_t,a_t)-s_{t+1}\right\|^2
\quad \text{（；：； 3.8）}
$$

$$
r_t^{\text{RND}}=\left\|\hat{\phi}(s_t)-\phi(s_t)\right\|^2
\quad \text{（RND ；：； 3.8）}
$$

$$
r_t^{\text{total}}=r_t^{\text{extrinsic}}+\beta r_t^{\text{intrinsic}}
\quad \text{（；：； 3.8）}
$$

## 

（）。、，$n$ 。

### 

， $n=|\mathcal{S}|$ ，
$n_A=|\mathcal{A}|$ 。

|                  |             |                                                                    |
| -------------------- | --------------- | ---------------------------------------------------------------------- |
| $\boldsymbol{v}_\pi$ | $n \times 1$    |                                                          |
| $\boldsymbol{r}_\pi$ | $n \times 1$    |                                                  |
| $P_\pi$              | $n \times n$    | ，$P_\pi[i,j]=\sum_a \pi(a\mid s_i)p(s_j\mid s_i,a)$ |
| $\boldsymbol{q}_\pi$ | $nn_A \times 1$ |  $(s,a)$  Q                                                  |
| $P$                  | $nn_A \times n$ | ，$P[(s,a),s']=P(s'\mid s,a)$                                  |
| $\Pi_\pi$            | $n \times nn_A$ | ，$\Pi_\pi[s,(s,a)]=\pi(a\mid s)$                              |

### 

****

：

$$
V^\pi(s)=\sum_a\pi(a\mid s)\left[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^\pi(s')\right]
$$

：

$$
\boldsymbol{v}_\pi
=
\boldsymbol{r}_\pi+\gamma P_\pi\boldsymbol{v}_\pi
$$

****

：

$$
V^*(s)=\max_a\left[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^*(s')\right]
$$

：

$$
\boldsymbol{v}_*
=
\boldsymbol{r}_*+\gamma P_*\boldsymbol{v}_*
\quad\text{（ max）}
$$

****

：

$$
\boldsymbol{v}=(I-\gamma P)^{-1}\boldsymbol{r}
$$

**V-Q **

：

$$
V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a)
$$

：

$$
\boldsymbol{v}_\pi=\Pi_\pi\boldsymbol{q}_\pi
$$

**Q **

：

$$
Q^\pi(s,a)
=
R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\sum_{a'}\pi(a'\mid s')Q^\pi(s',a')
$$

：

$$
\boldsymbol{q}_\pi
=
\boldsymbol{r}+\gamma P\Pi_\pi\boldsymbol{q}_\pi
$$

**Q **

：

$$
Q^*(s,a)
=
R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\max_{a'}Q^*(s',a')
$$

：

$$
\boldsymbol{q}_*
=
\boldsymbol{r}+\gamma P\cdot\mathrm{rowmax}(\boldsymbol{q}_*)
$$

**DP **

：

$$
V(s)\leftarrow
\sum_a\pi(a\mid s)
\left[
R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V(s')
\right]
$$

：

$$
\boldsymbol{v}_{k+1}
=
\boldsymbol{r}_\pi+\gamma P_\pi\boldsymbol{v}_k
$$

MC  TD ，。

###  Q  V

 $\boldsymbol{v}_\pi = \Pi_\pi \boldsymbol{q}_\pi$  $\boldsymbol{q}_\pi = \boldsymbol{r} + \gamma P \boldsymbol{v}_\pi$， $\Pi_\pi$：

$$
\Pi_\pi \boldsymbol{q}_\pi = \Pi_\pi \boldsymbol{r} + \gamma \Pi_\pi P \boldsymbol{v}_\pi
\quad\Longrightarrow\quad
\boldsymbol{v}_\pi = \underbrace{\Pi_\pi \boldsymbol{r}}_{\boldsymbol{r}_\pi} + \gamma \underbrace{\Pi_\pi P}_{P_\pi} \boldsymbol{v}_\pi
$$

Q （ $\Pi_\pi$ ），V  $\boldsymbol{r}_\pi$  $P_\pi$——"$Q$  $V$ "。

## 

，。

|      |                                      |                                                        |
| -------- | -------------------------------------------- | -------------------------------------------------------------- |
|  | 、、？           | $\mathcal{M}=\langle\mathcal{S},\mathcal{A},P,R,\gamma\rangle$ |
|  | ？     | $G_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k}$                      |
|  | ？                 | $\pi(s)$、$\pi(a\mid s)$、$\pi_\theta(a\mid s)$                |
|  | ？                 | $V^\pi(s)=\mathbb{E}_\pi[G_t\mid s_t=s]$                       |
|  | ？         | 、                                 |
|  | ？ | DP、MC、TD、$\delta$                                           |
|  | ？           | $Q^\pi(s,a)$、$Q^*(s,a)$                                       |
|  | ？       | Q-Learning、TD Target、$\varepsilon$-greedy                    |
|  | ？                 | $J(\theta)$、$\nabla_\theta J(\theta)$                         |
|  | ？               | $R(s,a)$、$F(s,a,s')$、$r_t^{\text{total}}$                    |

：，； DP、MC、TD ；；。

## 

### 

 3 。：

$$
G_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k}
$$

：

$$
G_t=r_t+\gamma G_{t+1}
$$

。 $V^\pi(s)$。

### 

 $P$  $R$ ， DP 。，：

- MC  $G_t$ ，。
- TD  $r+\gamma V(s')$ ，。
- TD Error $\delta=r+\gamma V(s')-V(s)$ 。

 Critic、DQN target、GAE 。

### 

$V^\pi(s)$ ， $s$ 。， 3.5 ：

$$
Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid s_t=s,a_t=a]
$$

， $\pi$ 。，$Q$  $V$ 。 $Q^*(s,a)$ ， $\arg\max_a Q^*(s,a)$ 。

###  Q-Learning

 3.5  TD 。 $(s,a,r,s')$  TD ：

$$
r+\gamma\max_{a'}Q(s',a')
$$

 $Q(s,a)$。，， Q 。 Q-Learning ；， 4 。

### 

 3.6 ：， $\pi_\theta(a\mid s)$，：

$$
J(\theta)=\mathbb{E}_{\pi_\theta}[G_t]
$$

，：$\nabla_\theta\log\pi_\theta(a\mid s)$ ，$G_t$ 。 5 。

### 

、。；。 3.8 ，。

## 

，：

1. ， MDP ？

::: details 
 $\mathcal{M}=\langle\mathcal{S},\mathcal{A},P,R,\gamma\rangle$。，$\mathcal{S}$ ，$\mathcal{A}$ ，$P(s'\mid s,a)$ ，$R(s,a)$  $R(s,a,s')$ ，$\gamma$ 。 MDP ，，。
:::

2.  RL ，？

::: details 
。，，。，。

$$
G_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k}
$$

， $\gamma$ 。，$\gamma<1$ 。
:::

3. $G_t$、$V^\pi(s)$、$Q^\pi(s,a)$、$J(\theta)$ ？

::: details 
$G_t$  $t$ 。$V^\pi(s)$  $s$  $\pi$ ，$G_t$ ，。$Q^\pi(s,a)$  $s$  $a$， $\pi$ ，$G_t$ ，。$J(\theta)$  $\pi_\theta$ ，。
:::

4. ？

::: details 
 $\pi$， $\pi(a\mid s)$ ：

$$
V^\pi(s)=\sum_a\pi(a\mid s)\left[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^\pi(s')\right].
$$

，，：

$$
V^*(s)=\max_a\left[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^*(s')\right].
$$

“”，“”。
:::

5. DP、MC、TD ，，？

::: details 
DP  $P$  $R$，，。MC ， episode  $G_t$ ；，，。TD ， episode ， $r+\gamma V(s')$ ；， $V(s')$ ，。
:::

6. TD Error  Critic、 Q  GAE ？

::: details 
TD Error

$$
\delta=r+\gamma V(s')-V(s)
$$

。Critic ； Q  Q ；GAE  TD Error ，。，TD Error 。
:::

7.  $Q(s,a)$ ？

::: details 
$Q(s,a)$  $s$  $a$ 。， $Q$ 。 $Q^*(s,a)$ ，

$$
\pi^*(s)=\arg\max_a Q^*(s,a).
$$

，$Q$ ，。
:::

8.  $J(\theta)$？

::: details 
 $\pi_\theta(a\mid s)$  $\theta$。， $\theta$ ：

$$
J(\theta)=\mathbb{E}_{\pi_\theta}[G_t].
$$

$J(\theta)$ 。 $\nabla_\theta J(\theta)$ ，，。
:::

9. ，？

::: details 
，，。，，。，，。

$$
F(s,a,s')=\gamma\Phi(s')-\Phi(s)
$$

，。
:::

。，，。

## 

|              |                                                    |                                                     |
| -------------------- | -------------------------------------------------------------------- | ------------------------------------------------------- |
|  4  Q    | $Q(s,a)$、$Q^*(s,a)$、$\arg\max_a Q(s,a)$、TD Target                 | ， Q            |
|  5       | $\pi_\theta(a\mid s)$、$J(\theta)$、$\nabla_\theta J(\theta)$、$G_t$ | ，                  |
|  6  Actor-Critic | $V(s)$、TD Error、$J(\theta)$                                        |  Critic，         |
|  7  PPO          | $V(s)$、、TD Error、                                 |  Critic ，                  |
|  8  RL | 、、、                                           |  token ， |

， 3 ，。，、、、 KL 。

## 

 3 ：

1.  MDP 。
2.  $G_t$ 。
3.  $V^\pi(s)$  $Q^\pi(s,a)$ 。
4. 。
5.  DP、MC、TD 。
6.  $J(\theta)$ 。
7. （On/Off-policy, Online/Offline）。
8. ，。

 $Q(s,a)$ ，：[ 4 ： Q ](../chapter04_dqn/intro)。
