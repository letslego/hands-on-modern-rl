# E.3.5 PG、DQN、GAE、PPO、GRPO 

> ****： E.3 ， [E.3.1](./calculus-basics)  [E.3.4](./calculus-derivations) 。

、、PPO  GRPO 。， DQN  GAE 。——。

---

## 

：

$$
\nabla_\theta J(\theta) \approx G_t\nabla_\theta\log\pi_\theta(a_t\mid s_t).
$$

：

$$
\nabla_\theta J(\theta)
=\sum_s d^\pi(s)\sum_a q_\pi(s,a)\nabla_\theta\pi_\theta(a\mid s).
$$

：

- $d^\pi(s)$  $\pi$  $s$ （"， $s$"）。
- $q_\pi(s,a)$ ， $s$  $a$ ，。
- $\nabla_\theta\pi_\theta(a\mid s)$  $\theta$ ， $a$ 。

， log 。：

$$
\nabla_\theta\pi_\theta(a\mid s)
=\pi_\theta(a\mid s)\nabla_\theta\log\pi_\theta(a\mid s),
$$

：

$$
\nabla_\theta J(\theta)
=\mathbb{E}_{s\sim d^\pi,a\sim\pi}
\left[q_\pi(s,a)\nabla_\theta\log\pi_\theta(a\mid s)\right].
$$

 $G_t$  $q_\pi(s_t,a_t)$， REINFORCE：

$$
\nabla_\theta J(\theta)
\approx
G_t\nabla_\theta\log\pi_\theta(a_t\mid s_t).
$$

，：

$$
\nabla_\theta J(\theta)
=\mathbb{E}
\left[A_\pi(s,a)\nabla_\theta\log\pi_\theta(a\mid s)\right].
$$

，——”、”。

---

## 

””，””—— Critic  DQN 。**？**  $\hat{A}_t$  $V(s)$ ；，。：。

 $(s_t,a_t,r_{t+1},s_{t+1})$，DQN  TD ：

$$
y_t=r_{t+1}+\gamma\max_{a'}Q_{\theta^-}(s_{t+1},a').
$$

 $\theta^-$ 。：

$$
L(\theta)=\frac{1}{2}\left(Q_\theta(s_t,a_t)-y_t\right)^2.
$$

：

$$
\nabla_\theta L(\theta)
=\left(Q_\theta(s_t,a_t)-y_t\right)
\nabla_\theta Q_\theta(s_t,a_t).
$$

， TD ：

$$
\delta_t=y_t-Q_\theta(s_t,a_t).
$$

 $\nabla_\theta Q_\theta(s_t,a_t)$ 。DQN 。

---

## GAE： TD 

 $\hat{A}_t$ ""，。：——（）；（TD）" + "——（，）。**GAE（Generalized Advantage Estimation）**—— TD ， $\lambda$ " MC  TD"。 TD ：

$$
\delta_t=R_{t+1}+\gamma V(s_{t+1})-V(s_t).
$$

 $\delta_t>0$， Critic ； $\delta_t<0$，。TD ，GAE  TD ：

$$
\hat{A}^{GAE}_t
=\delta_t+(\gamma\lambda)\delta_{t+1}+(\gamma\lambda)^2\delta_{t+2}+\cdots.
$$

 $\lambda\in[0,1]$ ：

- $\lambda$ ： TD ，，。
- $\lambda$ ：，，。

PPO  GAE，。

---

## PPO 

：

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{old}(a_t\mid s_t)}.
$$

PPO ：

$$
L^{CLIP}(\theta)=
\mathbb{E}_t\left[
\min\left(
 r_t(\theta)\hat{A}_t,
 \mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t
\right)
\right].
$$

，。

 $\hat{A}_t>0$，。， $1+\epsilon$ 。

 $\hat{A}_t<0$，。， $1-\epsilon$ 。

 `min`  `clip` ：**，**。

---

## GRPO 

GRPO 。 $n$ ，：

$$
r_1,r_2,\ldots,r_n.
$$

：

$$
\mu=\frac{1}{n}\sum_{i=1}^n r_i.
$$

：

$$
\sigma=\sqrt{\frac{1}{n}\sum_{i=1}^n(r_i-\mu)^2}.
$$

：

$$
\hat{A}_i=\frac{r_i-\mu}{\sigma+\epsilon}.
$$

 $[2,4,10]$， $5.33$。，；，。： Critic ，。

---

## 

 E.3 ：

|          |                                                              |                        |
| ------------ | ---------------------------------------------------------------------- | -------------------------- |
|  | $\nabla_\theta J=\mathbb{E}[\nabla\log\pi\cdot Q^\pi]$                 |          |
| DQN      | $L=\frac{1}{2}(Q_\theta-y_t)^2$                                        |          |
| GAE          | $\hat{A}^{GAE}_t=\sum_l(\gamma\lambda)^l\delta_{t+l}$                  | -    |
| PPO      | $\min(r_t\hat{A}_t,\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\hat{A}_t)$ |            |
| GRPO   | $\hat{A}_i=(r_i-\mu)/(\sigma+\epsilon)$                                |  Critic  |

。。

> ****：[E.3.6 ](./calculus-formulas-exercises) —— ，。
