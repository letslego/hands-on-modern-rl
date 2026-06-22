# E.3.6 

> ****： E.3 ， [E.3.1](./calculus-basics)  [E.3.5](./calculus-advanced-formulas) 。，。

 PPO  GRPO ，。，，，。

---

## 

|              |                                                                                    |                    |
| ---------------- | -------------------------------------------------------------------------------------- | ------------------------------ |
|          | $\theta \leftarrow \theta + \alpha \nabla J(\theta)$                                   |                  |
|          | $\theta \leftarrow \theta - \alpha \nabla L(\theta)$                                   |        |
|          | $\frac{dL}{d\theta}=\frac{dL}{dy}\frac{dy}{d\theta}$                                   |                  |
|          | $\nabla J \approx G_t\nabla\log\pi_\theta(a_t\mid s_t)$                                |          |
|      | $\nabla_\theta J=\sum_s d^\pi(s)\sum_a\nabla_\theta\pi_\theta(a\mid s)Q^\pi(s,a)$      |        |
|      | $\nabla_\theta\pi=\pi\nabla_\theta\log\pi$                                             |  |
|      | $`\nabla J \approx \hat{A}_t\nabla\log\pi_\theta(a_t\mid s_t)`$                          |        |
| PPO        | $r_t=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{old}(a_t\mid s_t)}$                           |                |
|  Taylor  | $f(\theta+\Delta)\approx f(\theta)+\nabla f^\top\Delta+\frac{1}{2}\Delta^\top H\Delta$ |                |
| PPO        | $\min(r_t\hat{A}_t,\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\hat{A}_t)$                 |                |
| GRPO       | $\hat{A}_i=\frac{r_i-\mu}{\sigma}$                                                     |  Critic      |

---

## ：

，：，""；；""；；PPO  Adam ； GRPO  Critic。，：、、。

---

## 

1. **。** ，，、 Adam 。。
2. **。** ，。PPO 。
3. ** Critic 。** Actor  Critic ；Critic ，。

::: warning 
，：(1) ，；(2) ，；(3)  Critic  Actor 。
:::

---

## 

1.  $J(\theta)=-(\theta-1)^2+2$ ， $\theta=0$、 $0.1$ 。
2.  $0.4$， $0.6$， $3$， PPO ？ $\epsilon=0.2$，？
3.  $V(s_t)=5$，$R_{t+1}=2$，$V(s_{t+1})=6$，$\gamma=0.9$， TD ？
