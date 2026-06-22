# E.4.3 PPO、RLHF  DPO 

> ****：[E.4.1 ](./information-basics)[E.4.2  KL](./information-cross-entropy-kl)——、 KL 。

---

、 KL 。。" KL ""RLHF ""DPO "，。

## PPO  RLHF  KL 

，，（reward hacking）。，PPO  RLHF  KL —— KL （），。

。 token  $0.01$， $0.20$。：

$$
\frac{0.20}{0.01}=20.
$$

 token  20 。 token ，——，（ reward hacking）。

，PPO  RLHF  KL ：

$$
\text{} = \text{} - \beta D_{KL}(\pi_{new}\|\pi_{old}).
$$

 $\beta$ 。，KL ，，。

 RLHF ，KL ：。，——、""。

---

## 、 KL 

，、 KL 。，——，、、。

$$
D_{KL}(P\|Q)=H(P,Q)-H(P).
$$

：**KL " $Q$  $P$ "，" $P$ "——，**。

。：

$$
H(P,Q)=-\sum_x P(x)\log Q(x),
$$

$$
H(P)=-\sum_x P(x)\log P(x),
$$

：

$$
H(P,Q)-H(P)
=
-\sum_xP(x)\log Q(x)
+\sum_xP(x)\log P(x).
$$

：

$$
\sum_xP(x)\log\frac{P(x)}{Q(x)}
=D_{KL}(P\|Q).
$$

。 $P$ ， $H(P,Q)$  $D_{KL}(P\|Q)$ 。、、——，。

---

## DPO：

RLHF ：， PPO 。——（、、、Critic）， KL 。DPO（Direct Preference Optimization）：** PPO，（" A  B "）？** DPO  KL ， RLHF 。

：

$$
\mathcal{L}_{DPO}
=
-\mathbb{E}\left[
\log\sigma\left(
\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}
-
\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}
\right)
\right].
$$

。——：

$$
\log\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}.
$$

：， $y$。。 $y$：

|                            |    |
| ------------------------------ | ------ |
|  $\pi_\theta(y\mid x)$ | $0.20$ |
|  $\pi_{ref}(y\mid x)$  | $0.05$ |

 $\frac{0.20}{0.05}=4$， $\log 4$。。

DPO  winner（$y_w$） loser（$y_l$）：

$$
\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}
-
\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}.
$$

 winner 、loser ，，——”、”。

$\sigma$  sigmoid ， $(0,1)$ ，$\log\sigma$ 。”，”。

DPO  KL ： $\pi_{ref}$ ，””。，。， RLHF ：

$$
J(\pi)=\mathbb{E}_\pi[r(x,y)]-\beta D_{KL}(\pi_\theta\|\pi_{ref})
$$

：**，**。

---

## 

：

|            |                  | /                                        |  RL            |
| -------------- | ---------------------------- | ---------------------------------------------------- | ------------------------ |
| KL         |            |  $- \beta D_{KL}$                                | PPO/RLHF   |
| -KL  |      | $D_{KL}(P\|Q)=H(P,Q)-H(P)$                           |  =  KL |
| DPO  |  | $\log\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}$ |    |

：KL  ->  KL  -> DPO  KL ，。

> ****：[E.4.4 ](./information-mutual-info) —— 。
