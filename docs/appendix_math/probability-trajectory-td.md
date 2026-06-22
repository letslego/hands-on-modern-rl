# E.2.4 、Baseline  GAE

> ****：[E.2.1 ](./probability-basics)[E.2.2 ](./probability-value)——、。

---

 $\pi(a\mid s)$  $p(s'\mid s,a)$。——，。 RL ：，，，。""，，——。

：

$$
\tau=(s_0,a_0,s_1,a_1,s_2).
$$

：

1.  $p(s_0)$。
2.  $\pi(a_t\mid s_t)$。
3.  $p(s_{t+1}\mid s_t,a_t)$。

：

$$
p(\tau\mid\pi)=p(s_0)\prod_{t=0}^{T-1}\pi(a_t\mid s_t)p(s_{t+1}\mid s_t,a_t).
$$

：

|                                   |    |
| ----------------------------------- | ------ |
|  $p(s_0)$               | $0.6$  |
|  $\pi(a_0\mid s_0)$   | $0.5$  |
|  $p(s_1\mid s_0,a_0)$ | $0.8$  |
|  $\pi(a_1\mid s_1)$   | $0.25$ |
|  $p(s_2\mid s_1,a_1)$ | $0.4$  |

：

$$
0.6\times0.5\times0.8\times0.25\times0.4=0.024.
$$

？””：

$$
J(\theta)=\mathbb{E}_{\tau\sim p_\theta(\tau)}[G(\tau)]=\sum_{\tau}p_\theta(\tau)G(\tau).
$$

 $G(\tau)$ 。，。

---

## Baseline 

 $G_t$ 。—— $100$， $-5$。，，。：””，””—— $G_t$  $G_t-b(s_t)$。 $b(s_t)$  **baseline**， $V(s_t)$。，：，？？：** baseline ，**。

：

$$
\mathbb{E}_{a\sim\pi(\cdot\mid s)}
\left[\nabla_\theta\log\pi_\theta(a\mid s)b(s)\right]=0.
$$

 $b(s)$ ：

$$
b(s)\sum_a\pi_\theta(a\mid s)\nabla_\theta\log\pi_\theta(a\mid s).
$$

：

$$
\nabla_\theta\log\pi_\theta(a\mid s)
=\frac{\nabla_\theta\pi_\theta(a\mid s)}{\pi_\theta(a\mid s)}.
$$

：

$$
b(s)\sum_a\pi_\theta(a\mid s)
\frac{\nabla_\theta\pi_\theta(a\mid s)}{\pi_\theta(a\mid s)}
=b(s)\sum_a\nabla_\theta\pi_\theta(a\mid s).
$$

：

$$
b(s)\nabla_\theta\sum_a\pi_\theta(a\mid s).
$$

 $1$：

$$
\sum_a\pi_\theta(a\mid s)=1.
$$

：

$$
b(s)\nabla_\theta 1=0.
$$

 baseline ，。：

$$
\nabla_\theta J(\theta)
=\mathbb{E}_\pi\left[
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
(G_t-b(s_t))
\right].
$$

 $b(s_t)=V^\pi(s_t)$ ，：

$$
A^\pi(s_t,a_t)=G_t-V^\pi(s_t).
$$

---

##  TD error  GAE

 baseline ，：$G_t - b(s_t)$  $G_t$ ？——（，）。（TD）" + "——（，）。GAE（Generalized Advantage Estimation）：（），（）。

 TD error—— GAE 。TD error """"：

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t).
$$

。：

$$
r_t=2,\qquad \gamma=0.9,\qquad V(s_{t+1})=5,\qquad V(s_t)=6.
$$

：

$$
\delta_t=2+0.9\times5-6=0.5.
$$

：“ + ” $0.5$，。

 TD error ，；，。GAE  TD error ：

$$
\hat{A}_t^{GAE(\gamma,\lambda)}
=\sum_{k=0}^{T-t-1}(\gamma\lambda)^k\delta_{t+k}.
$$

 $\lambda=0$，：

$$
\hat{A}_t=\delta_t.
$$

 $\lambda$  $1$， TD error ，。 $\lambda$ “-”：

| $\lambda$ |  |              |
| --------- | -------- | ---------------- |
| $0$       |  TD  | ， |
| $1$       |  | ， |
| $0.95$    |      | PPO        |

---

## PPO 

，：""，baseline ，GAE  TD  MC 。 PPO 。PPO ：**，？** ""，；GAE 。PPO ：。

PPO ：

$$
L^{CLIP}(\theta)
=
\mathbb{E}\left[
\min\left(
r_t(\theta)\hat{A}_t,\,
\mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t
\right)
\right],
$$

：

$$
r_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{old}}(a_t\mid s_t)}.
$$

。，，” / ”。

。 $0.2$， $0.3$：

$$
r_t=\frac{0.3}{0.2}=1.5.
$$

 $\hat{A}_t=4$，：

$$
r_t\hat{A}_t=1.5\times4=6.
$$

 $\epsilon=0.2$， $[0.8,1.2]$，：

$$
\mathrm{clip}(1.5,0.8,1.2)\times4=4.8.
$$

：

$$
\min(6,4.8)=4.8.
$$

 PPO ，：

1. 。
2. 。
3. 。

---

## 

 PPO ：

|      |                      |                            |
| -------- | -------------------------------- | ---------------------------------- |
|  |        |        |
| Baseline | ，     |  baseline ， |
| GAE      | MC 、TD ，   |  TD error      |
| PPO  |  |        |

""；baseline ""；GAE ；PPO 、。， RL 。

> ****：[E.2.5 ](./probability-bellman-advanced) —— ，。
