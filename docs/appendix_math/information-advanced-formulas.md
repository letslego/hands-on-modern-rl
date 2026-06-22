# E.4.5 KL、RLHF、DPO 

> ****： E.4 ， [E.4.1](./information-basics)  [E.4.4](./information-mutual-info) 。

---

 E.4 ，。。

## KL、

、 KL 。——。

$$
D_{KL}(P\|Q)=H(P,Q)-H(P).
$$

：

$$
H(P,Q)-H(P)
= -\sum_x P(x)\log Q(x) + \sum_x P(x)\log P(x).
$$

：

$$
=\sum_x P(x)\log\frac{P(x)}{Q(x)}
=D_{KL}(P\|Q).
$$

 KL ： $P$， $Q$ ，。

：

-  $H(P,Q)$。
-  $P$ ， $D_{KL}(P\|Q)$。

、、。

---

## ：RLHF  KL 

 KL  RLHF ——，KL 。

RLHF ：

$$
\max_\pi \; \mathbb{E}_{x,y\sim\pi}[r(x,y)]
-\beta D_{KL}(\pi(\cdot\mid x)\|\pi_{ref}(\cdot\mid x)).
$$

：

- $r(x,y)$  $y$ 。
- $\pi$ 。
- $\pi_{ref}$ ， SFT 。
- $\beta$ ””””。

 $\beta$ ，， reward hacking； $\beta$ ，，。

---

## ：DPO 

DPO  PPO，。——。

$$
\log\frac{\pi_\theta(y\mid x)}{\pi_{ref}(y\mid x)}.
$$

 $(x,y_w,y_l)$， $y_w$ ，$y_l$ ，DPO ：

$$
\mathcal{L}_{DPO}(\theta)
=-\log\sigma\left(
\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}
-
\log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}
\right]
\right).
$$

：

-  winner ，。
-  loser ，，。
- ，。

DPO ” winner ”，”，winner  loser ”。 KL 。

---

## ：

 KL ，””——。

$$
I(X;Y)=D_{KL}(P_{XY}\|P_XP_Y)=H(X)-H(X\mid Y).
$$

， $\phi(s)$  $G_t$ ：

$$
I(\phi(s);G_t) \text{ }.
$$

。，，。

，、、 RL 。

---

## 

 E.4 ：

|      | /                                                                                                |                            |
| ------------ | -------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| KL-- | $D_{KL}(P\|Q)=H(P,Q)-H(P)$                                                                                     |        |
| RLHF     | $\max_\pi \mathbb{E}[r]-\beta D_{KL}(\pi\|\pi_{ref})$                                                          |          |
| DPO      | $-\log\sigma(\beta\log\frac{\pi_\theta(y_w)}{\pi_{ref}(y_w)}-\beta\log\frac{\pi_\theta(y_l)}{\pi_{ref}(y_l)})$ |                  |
|        | $I(X;Y)=H(X)-H(X\mid Y)=D_{KL}(P_{XY}\|P_XP_Y)$                                                                |  $Y$  $X$  |

> ****：[E.4.6 ](./information-formulas-exercises) —— ，。
