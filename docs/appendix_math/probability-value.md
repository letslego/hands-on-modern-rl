# E.2.2 

> ****：[E.2.1 、](./probability-basics)——。

---

——""。：，，，""。

## 

 $s$ ，：

|  |   |  |
| ---- | ----- | -------- |
| A    | $0.5$ | $8$      |
| B    | $0.3$ | $4$      |
| C    | $0.2$ | $-2$     |

 $\pi$  $s$ ？——：

$$
v_\pi(s)=0.5\times8+0.3\times4+0.2\times(-2)=4.8.
$$



$$
v_\pi(s)=\mathbb{E}_\pi[G_t \mid S_t=s]
$$

。$\mathbb{E}_\pi$  $\pi$ " $\pi$ "， $\mid S_t=s$ " $s$ "。：** $s$ ， $\pi$ ，**。

 $v_\pi(s)=4.8$ —— $8$、$4$  $-2$。$4.8$ 。""""：，。

### 

——。 RL ，：

$$
G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+\cdots.
$$

$\gamma$  $1$，""；$\gamma$ ，""。：

 $2, 1, 3, 0, 1$。

| $\gamma$ |  $G$                                               |                  |
| -------- | ---------------------------------------------------------- | -------------------- |
| $0.5$    | $2+0.5\times1+0.25\times3+0.125\times0+0.0625\times1=3.19$ |        |
| $0.9$    | $2+0.9+0.81\times3+0.729\times0+0.6561\times1=5.97$        |            |
| $0.99$   | $2+0.99+0.9801\times3+\cdots\approx6.92$                   |  |

$\gamma=0.5$ ， 5  $0.0625$，；$\gamma=0.99$ ， 5  $0.96$，。

：$\gamma$ ，（），（，）。

---

## ：

，。

 A ：

$$
4, \quad 5, \quad 6.
$$

 B ：

$$
0, \quad 5, \quad 10.
$$

 $5$。 B —— $0$， $10$。， B ，。。

 B ， $-5, 0, 5$。：

$$
\mathrm{Var}(G)=\frac{(-5)^2+0^2+5^2}{3}=\frac{50}{3}=16.67.
$$

$\mathrm{Var}$ （variance）。 $\mathrm{Var}(X) = \mathbb{E}[(X-\mathbb{E}[X])^2]$——，。

，。 baseline、advantage、GAE ——。

### 

。" right" $\hat{A}\cdot\nabla\log\pi$， $\hat{A}$ 。

** A**（ $2$ ）：，。

** B**（ $-8$  $+12$ ）："right ，"，"right ，"。，。

 RL 。 baseline、GAE、PPO ，：**？**

---

## 

：

|      |                                                     | RL                                  |
| -------- | ------------------------------------------------------- | --------------------------------------- |
|  |  $v_\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s]$ | ""        |
|  | $G_t=\sum_{k=0}\gamma^k R_{t+k+1}$                      | ，$\gamma$  |
|      | $\mathrm{Var}(X)=\mathbb{E}[(X-\mathbb{E}[X])^2]$       | ，      |

——""。""——$\gamma$ ，$\gamma$ 。：，。、baseline、GAE ，。

> ****：[E.2.3 、](./probability-sampling) —— ，。
