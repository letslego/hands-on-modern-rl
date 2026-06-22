# E.2.5 

> ****：[E.2.1 ](./probability-basics)[E.2.2 ](./probability-value)——。

---

""：

$$
v_\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s].
$$

， $\mathbb{E}_\pi$ 。、，——" 85 "、，。，。****——””” + ”，。

 $G_t$ ” + ”：

$$
G_t=R_{t+1}+\gamma G_{t+1}.
$$

：

$$
v_\pi(s)=\mathbb{E}_\pi[R_{t+1}+\gamma G_{t+1}\mid S_t=s].
$$

——：。：

$$
v_\pi(s)=\mathbb{E}_\pi[R_{t+1}\mid S_t=s]+\gamma\mathbb{E}_\pi[G_{t+1}\mid S_t=s].
$$

 $\mathbb{E}_\pi[R_{t+1}\mid S_t=s]$ ””，：

$$
\mathbb{E}_\pi[R_{t+1}\mid S_t=s]
=\sum_a \pi(a\mid s)\sum_r p(r\mid s,a)r.
$$

 $\mathbb{E}_\pi[G_{t+1}\mid S_t=s]$ ""：

$$
\mathbb{E}_\pi[G_{t+1}\mid S_t=s]
=\sum_a \pi(a\mid s)\sum_{s'}p(s'\mid s,a)v_\pi(s').
$$

，：

$$
v_\pi(s)=\sum_a\pi(a\mid s)
\left[
\sum_r p(r\mid s,a)r
+\gamma\sum_{s'}p(s'\mid s,a)v_\pi(s')
\right].
$$

，，""：

- $\sum_a\pi(a\mid s)$：。
- $\sum_r p(r\mid s,a)r$：。
- $\sum_{s'}p(s'\mid s,a)v_\pi(s')$：。

### 

。—— $s_1$ ， $s_2$， $2$； $s_2$ ， $s_1$， $1$。$\gamma=0.5$。

，$\pi(a\mid s)$  $1$、 $0$，""。 $s_1$ ：

$$
v(s_1)=\underbrace{2}_{\text{}}+\gamma\underbrace{(1\times v(s_2))}_{\text{ }s_2\text{， }1}=2+0.5v(s_2).
$$

 $s_2$ ：

$$
v(s_2)=\underbrace{1}_{\text{}}+\gamma\underbrace{(1\times v(s_1))}_{\text{ }s_1\text{， }1}=1+0.5v(s_1).
$$

，" + "。

### ？

 $s_1$ ， $\pi(\text{left}\mid s_1)=0.3$、$\pi(\text{right}\mid s_1)=0.7$：

-  left  $1$， $s_1$。
-  right  $3$， $s_2$。

 $s_1$ ：

$$
v(s_1)=0.3\times[1+0.5v(s_1)]+0.7\times[3+0.5v(s_2)].
$$

" + "，。****。

---

## 

 $v_\pi(s)$ " $s$ ，"。，： $s$ ， left ， right ？""——、。，****。

$$
q_\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a].
$$

 $q_\pi(s,a)$ ： $s$  $a$， $\pi$ ，。

：

$$
q_\pi(s,a)=\sum_r p(r\mid s,a)r
+\gamma\sum_{s'}p(s'\mid s,a)v_\pi(s').
$$

 $q$  $v$ ：——

$$
v_\pi(s)=\sum_a\pi(a\mid s)q_\pi(s,a).
$$

，” $s$ ”” × ”。

****：

$$
A_\pi(s,a)=q_\pi(s,a)-v_\pi(s).
$$

” $a$  $s$ ”。” 10、 8， 2”，。

###  q、v  A 

 $s$ ， $0.4$  left、$0.6$  right。：

|   |  $q(s,a)$ |
| ----- | ----------------- |
| left  | $5$               |
| right | $8$               |

：

$$
v(s)=0.4\times5+0.6\times8=2+4.8=6.8.
$$

：

$$
A(s,\text{left})=5-6.8=-1.8, \qquad A(s,\text{right})=8-6.8=1.2.
$$

left  $1.8$（），right  $1.2$（）。：right ，left 。

---

## 

：

$$
\rho_t=\frac{\pi(a_t\mid s_t)}{b(a_t\mid s_t)}.
$$

 $b$ ， $\pi$ ，：

$$
\rho_{0:T}=\prod_{t=0}^{T}\frac{\pi(a_t\mid s_t)}{b(a_t\mid s_t)}.
$$

 $\prod$（ $\pi$）""， $\sum$ ""。：

$$
\hat{v}_\pi(s)=\frac{1}{N}\sum_{i=1}^N \rho^{(i)}G^{(i)}.
$$

，：，，。、。

### ：

，：

|   | $\pi(a_t\mid s_t)$ | $b(a_t\mid s_t)$ |     |
| ----- | ------------------ | ---------------- | ----------- |
| $t=0$ | $0.6$              | $0.3$            | $0.6/0.3=2$ |
| $t=1$ | $0.8$              | $0.4$            | $0.8/0.4=2$ |

：

$$
\rho_{0:2}=2\times2=4.
$$

 $G=5$， $4\times5=20$。。 $3$， $10$  $3^{10}=59049$——""。PPO 。

---

## 

****：

$$
\mathrm{Cov}(X,Y)=\mathbb{E}[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])].
$$

 $[-1,1]$：

$$
\rho_{X,Y}=\frac{\mathrm{Cov}(X,Y)}{\sigma_X\sigma_Y}.
$$

，。：

$$
g_t=\hat{A}_t\nabla_\theta\log\pi_\theta(a_t\mid s_t).
$$

 $\hat{A}_t$ ，$g_t$ 。 baseline、advantage 、GAE，。

### ：

， $\hat{A}=[2, -1]$， $[4, 1]$。

：$\bar{A}=0.5$，$\bar{g}=2.5$。

：

$$
\mathrm{Cov}=\frac{(2-0.5)(4-2.5)+(-1-0.5)(1-2.5)}{2}=\frac{1.5\times1.5+(-1.5)\times(-1.5)}{2}=\frac{4.5}{2}=2.25.
$$

，——。 $0$，，。

---

## 

""，：

|            |                                                                                       |                                      |
| -------------- | ----------------------------------------------------------------------------------------- | ---------------------------------------- |
|  | $v_\pi(s)=\sum_a\pi(a\mid s)[\sum_r p(r\mid s,a)r+\gamma\sum_{s'}p(s'\mid s,a)v_\pi(s')]$ | """ + "        |
|        | $q_\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a]$                                          | ""         |
|        | $A_\pi(s,a)=q_\pi(s,a)-v_\pi(s)$                                                          | ""           |
|  | $\rho_{0:T}=\prod_t\frac{\pi(a_t\mid s_t)}{b(a_t\mid s_t)}$                               | ， |
|          | $\mathrm{Cov}(X,Y)=\mathbb{E}[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])]$                        | ， |

：，" + "。，。，。

> ****：[E.2.6 ](./probability-formulas-exercises) —— ，。
