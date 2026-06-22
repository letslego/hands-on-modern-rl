# E.3.4 、Taylor  GRPO 

> ****：[E.3.2 ](./calculus-policy-gradient)——。

---

## 

，。 $\pi_\theta(a\mid s)$ ——$\pi$  softmax ，。**** $\nabla_\theta \pi$  $\pi \cdot \nabla_\theta \log \pi$，，。：

$$
\nabla_\theta J(\theta)
=\mathbb{E}_\pi[
G_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)
].
$$

 $\mathbb{E}_\pi[\cdot]$ ” $\pi$ ”，-，。：

$$
\nabla_\theta \log \pi_\theta(a\mid s)
=
\frac{\nabla_\theta \pi_\theta(a\mid s)}
{\pi_\theta(a\mid s)}.
$$

 $\pi_\theta(a\mid s)$，：

$$
\nabla_\theta \pi_\theta(a\mid s)
=
\pi_\theta(a\mid s)\nabla_\theta\log\pi_\theta(a\mid s).
$$

： $\pi_\theta$ ， $\log\pi_\theta$ 。。，：

$$
J(\theta)=\sum_a \pi_\theta(a\mid s)Q^\pi(s,a).
$$

，$Q^\pi(s,a)$  $\theta$， $\pi_\theta$  $\theta$：

$$
\nabla_\theta J(\theta)
=\sum_a \nabla_\theta\pi_\theta(a\mid s)Q^\pi(s,a).
$$

， $\nabla_\theta\pi_\theta$ ：

$$
\nabla_\theta J(\theta)
=\sum_a
\pi_\theta(a\mid s)
\nabla_\theta\log\pi_\theta(a\mid s)
Q^\pi(s,a).
$$

： $\pi_\theta(a\mid s)$ ，""——：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{a\sim\pi_\theta(\cdot\mid s)}
[
\nabla_\theta\log\pi_\theta(a\mid s)Q^\pi(s,a)
].
$$

。（ $d^\pi(s)$  $\pi$  $s$ ），：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_\pi[
\nabla_\theta\log\pi_\theta(a_t\mid s_t)Q^\pi(s_t,a_t)
].
$$

，$Q^\pi(s_t,a_t)$ ， $G_t$  $\hat{A}_t$ ：

$$
\nabla_\theta J(\theta)
\approx
\mathbb{E}_\pi[
\nabla_\theta\log\pi_\theta(a_t\mid s_t)\hat{A}_t
].
$$

 REINFORCE、Actor-Critic  PPO 。

---

## Taylor 、Hessian  PPO 

——""，。，：，。**Taylor **""。，（、）。PPO  TRPO ""，——Taylor 。

$$
f(x+h)\approx f(x)+f'(x)h.
$$

。：

$$
f(x)=x^2,\qquad x=3,\qquad h=0.1.
$$

：

$$
f(3.1)=9.61.
$$

：

$$
f(3)+f'(3)h=9+6\times0.1=9.6.
$$

， $0.01$。 Taylor ：

$$
f(x+h)\approx f(x)+f'(x)h+\frac{1}{2}f''(x)h^2.
$$

 $f(x)=x^2$，$f''(x)=2$，：

$$
9+6\times0.1+\frac{1}{2}\times2\times0.1^2=9.61.
$$

， $f''$  Hessian  $H$（）：

$$
f(\theta+\Delta\theta)
\approx
f(\theta)
\nabla f(\theta)^\top\Delta\theta
\frac{1}{2}\Delta\theta^\top H\Delta\theta.
$$

PPO  TRPO ""，，，——，。

 PPO ：

$$
r_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{old}}(a_t\mid s_t)},
$$

 $\theta_{old}$ ：

$$
r_t(\theta)
\approx
1
+\nabla_\theta r_t^\top(\theta-\theta_{old})
+\frac{1}{2}(\theta-\theta_{old})^\top
\nabla_\theta^2 r_t
(\theta-\theta_{old}).
$$

：

|      |                            |
| ------ | ------------------------------ |
| $1$    | ， $1$   |
|  |          |
|  | ， |

PPO  Hessian， $r_t(\theta)$ ，。

---

## GRPO 

、PPO  Taylor ， $\hat{A}_t$。（ PPO） Critic ， Critic ，。**GRPO ： Critic，""。** ：，，，""。， prompt  4 ，：

$$
r=[2,4,6,8].
$$

：

$$
\mu=\frac{2+4+6+8}{4}=5.
$$

：

$$
\sigma=
\sqrt{
\frac{(2-5)^2+(4-5)^2+(6-5)^2+(8-5)^2}{4}
}
=\sqrt{5}.
$$

 4 ：

$$
\hat{A}_4=\frac{8-5}{\sqrt{5}}\approx1.34.
$$

：

$$
\hat{A}_i=\frac{r_i-\mu}{\sigma}.
$$

：

1. ：。
2. ：——，，。

GRPO  PPO  Critic ， baseline。””，””。

---

## 

：

|           |                                          |                             |
| ------------- | ------------------------------------------------ | ------------------------------- |
|   | $\nabla\pi = \pi\nabla\log\pi$                   |  log  |
| Taylor    | $f(x+h)\approx f(x)+f'(x)h+\frac{1}{2}f''(x)h^2$ |  PPO  |
| GRPO  | $\hat{A}_i=(r_i-\mu)/\sigma$                     |  Critic       |

、、 Critic 。。

> ****：[E.3.5 ](./calculus-advanced-formulas) —— PG、DQN、GAE、PPO、GRPO 。
