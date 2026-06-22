# E.3.3 PPO 、Adam 

> ****：[E.3.2 ](./calculus-policy-gradient)——。

---

## PPO 

"，"，。，——，。PPO 。——（probability ratio）：

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{old}(a_t\mid s_t)}.
$$

。 $0.2$， $0.3$：

$$
r_t=\frac{0.3}{0.2}=1.5.
$$

 $1.5$ 。 $\hat{A}_t=4$，：

$$
r_t\hat{A}_t=1.5\times4=6.
$$

 PPO 。 $[0.8,1.2]$， $1.5$  $1.2$：

$$
\mathrm{clip}(r_t,0.8,1.2)\hat{A}_t=1.2\times4=4.8.
$$

PPO ，，。 min 。

---

## Adam：

， PPO 。，——，。，，。Adam ""。

PPO ， Adam 。，，。Adam ：

- ：，。
- ：，。

：，Adam ；、，Adam ，。

，，。 Adam、、， RL 。

---

## 

：

|      |      |                                        |
| -------- | ------------ | ---------------------------------------------- |
| PPO  |  | ，             |
| Adam     |  |  |

：PPO ，Adam 。 PPO 。

> ****：[E.3.4 ：log trick  Taylor](./calculus-derivations) ——  PPO 。
