# E.4.1 ：、

> ****：，[](./intro)[E.2.1 ](./probability-basics)。

---

## 

—— $s$，：

$$
\pi(\cdot\mid s).
$$

，：

1. ？（：）
2. ？（：）
3. ？（：KL ）

 DPO 。， RL 。

---

## ，

： 1  8 ，”/”。 3，？” 4 ？”” 2 ？”” 3 ？”—— 3 。 1  2，1 。

，，。：**，，””**。

，（self-information）””：

$$
I(x)=-\log_2 p(x).
$$

 $\log_2$  2 ，$p(x)$  $x$ 。（ $\log$  1 ）。

。 $p(x)=1/2$（）：

$$
I(x)=-\log_2(1/2)=1.
$$

 $p(x)=1/8$（）：

$$
I(x)=-\log_2(1/8)=3.
$$

 $1/2$  $1/8$， 1  3。””， KL 。

---

## ——””

，。””——。

（entropy， $H$ ）：

$$
H(P)=-\sum_x p(x)\log_2 p(x).
$$

 $\sum_x$ " $x$ "， $p(x)$ 。

。 A ：

|   |   |
| ----- | ----- |
| left  | $0.5$ |
| right | $0.5$ |

 B ：

|   |   |
| ----- | ----- |
| left  | $0.9$ |
| right | $0.1$ |

 A ：

$$
H(A)=-0.5\log_2 0.5-0.5\log_2 0.5=1.
$$

 B ：

$$
H(B)=-0.9\log_2 0.9-0.1\log_2 0.1\approx0.47.
$$

A ，""，；B ，，。

---

## ——""

，。，——""。，，。

：

$$
J(\pi)=\mathbb{E}[G]+\beta H(\pi).
$$

 $\mathbb{E}[G]$ （$\mathbb{E}$ ），$H(\pi)$ ，$\beta$ ，""。

。：

|  |  |     | $\beta=0.5$       |
| ---- | -------- | ----- | ------------------------- |
| A    | $10$     | $1.0$ | $10+0.5\times1=10.5$      |
| B    | $10.3$   | $0.1$ | $10.3+0.5\times0.1=10.35$ |

B ， A 。，A  B。

：，，。，。

---

## ：bit  nat

，。 $\log_2$（ 2 ）， bit； $\ln$（ $e \approx 2.718$ ）， nat。

：$1\text{ nat} = \log_2 e \approx 1.443\text{ bit}$。，、softmax、。

， $\log_2$ ：

$$
H=-0.5\log_2 0.5-0.5\log_2 0.5=1\text{ bit}.
$$

：

$$
H=-0.5\ln0.5-0.5\ln0.5\approx0.693\text{ nat}.
$$

，，—— 1  1000 。，。

---

## 

：？——。

 $[0.5,0.5]$，； $[0.99,0.01]$，。””：

$$
H([0.5,0.5])=1\text{ bit},
$$



$$
H([0.9,0.1])\approx0.47\text{ bit}.
$$

””，，。，。

---

##  softmax 

，。，，，。——softmax 。

， logits $z=[2,1]$。softmax —— 0  1 ， 1：

$$
\pi(a_i\mid s)=\frac{e^{z_i}}{\sum_j e^{z_j}}.
$$

 $e^{z_i}$ ， $\sum_j e^{z_j}$ 。：

$$
\pi(a_1\mid s)=\frac{e^2}{e^2+e^1}\approx0.73,
\qquad
\pi(a_2\mid s)\approx0.27.
$$

softmax 。、 KL ， softmax 。，、PPO  DPO （）。

---

## 

""， RL ：

|     |                    |                                 | RL                  |
| ------- | ------------------------------ | --------------------------------------- | --------------------------- |
|   | ""           | $I(x)=-\log p(x)$                       |       |
|       |              | $H(P)=-\sum_x p(x)\log p(x)$            | ，    |
|   |        | $J=\mathbb{E}[G]+\beta H(\pi)$          |     |
|   |            |                         | "" =  |
| softmax |  | $\pi(a_i\mid s)=e^{z_i}/\sum_j e^{z_j}$ |   |

""。—— vs 、 vs —— KL 。

> ****：[E.4.2  KL ](./information-cross-entropy-kl) —— 。
