# E.4.6 

> ****： E.4 ， [E.4.1](./information-basics)  [E.4.5](./information-advanced-formulas) 。，。

---

 E.4 ，。。

## 

|            |                                                                                                                                                                  |              |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
|          | $I(x)=-\log p(x)$                                                                                                                                                    |    |
|              | $H(P)=-\sum_x p(x)\log p(x)$                                                                                                                                         | 、     |
|          | $J=\mathbb{E}[G]+\beta H(\pi)$                                                                                                                                       | ，   |
|          | $H(P,Q)=-\sum_x P(x)\log Q(x)$                                                                                                                                       | 、   |
| KL         | $D_{KL}(P\|Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}$                                                                                                                      |          |
| -KL  | $D_{KL}(P\|Q)=H(P,Q)-H(P)$                                                                                                                                           | KL         |
| KL         | $\text{reward}-\beta D_{KL}$                                                                                                                                         | PPO/RLHF   |
| RLHF       | $J(\pi)=\mathbb{E}_\pi[r(x,y)]-\beta D_{KL}(\pi_\theta\|\pi_{ref})$                                                                                                  |  |
| DPO        | $\mathcal{L}_{DPO}=-\mathbb{E}[\log\sigma(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)})]$ |    |
|          | $I(X;Y)=H(X)-H(X\mid Y)$                                                                                                                                             |  |

---

## 

：””””，、KL、RLHF  DPO 。，：、，？

---

## 

1. **。** ，，。
2. ** KL 。** KL ，$D_{KL}(P\|Q)$  $D_{KL}(Q\|P)$ 。
3. ** KL 。**  RLHF ，KL 。

---

## 

1.  $[0.5,0.5]$  $[0.9,0.1]$，？？
2.  $[0.5,0.5]$， $[0.8,0.2]$， $D_{KL}(\pi_{old}\|\pi_{new})$ 。
3.  RLHF  $\mathbb{E}[r]-\beta D_{KL}$ ，$\beta$ ，？
