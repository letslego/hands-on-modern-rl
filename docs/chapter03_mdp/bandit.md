# 3.1 ：RL 

## 

****

- ，。
- 、、。
- **** RL ：，，。

 CartPole  DPO，：**””？** ””， CartPole 。

RL ：**（trial-and-error）**——，；**（delayed reward）**—— [^5]。， RL 。

，****：**（Exploration vs. Exploitation）**。，， 100 ，——。，**，？** ：，；，。

””（Multi-Armed Bandit, MAB）[^1][^2]，：（）、（）、（）。 RL ，””（，），”””-”。

CartPole ：，，，。””，：

1. **？** → 
2. **””””？** → 

::: info 
 RL **（distinctive challenge）**。、，RL ：**””**  **””**。，，。
:::

****

””，。：**””**  **””**。

$$
\mathbb{E}[R_a] = p_a \cdot (+1) + (1-p_a)\cdot(-1) = 2p_a - 1 \quad \text{（：）}
$$

> ** (Expected Reward of an Action)：**
>
> - $\mathbb{E}$：（Expectation），。
> - $R_a$： $a$（）。
> - $p_a$： $a$ （）。

$$
\mathbb{E}[R_T] = \mathbb{E}[R_{a_1}] + \mathbb{E}[R_{a_2}] + \cdots + \mathbb{E}[R_{a_T}] = \sum_{t=1}^{T} \mathbb{E}[R_{a_t}] \quad \text{（T ：）}
$$

> **T  (Expected Total Return over T steps)：**
>
> - $T$：。
> - $a_t$： $t$ 。
> - $R_T$： $T$ 。

（）：，。（$T$ ），。，””。

**（policy）**，：**，。** ，" A  B"； CartPole ，"，"；，"， token "。

## 

。 1 ， 2 （ +1），（ -1）。 100 。。

::: info （Episode）？
 RL ，**（Episode）**。 1  Game Over ， CartPole 。

：（，）。（CartPole、Atari）。，，****。
:::

， RL ：**，；，。** ，；，。

## ，

，：**A  60%，B  40%**。，，。

""，：**，，。**  100 ，""、" A"、""，。

，""——**（Expected Value）**。

::: details ？
**（Expected Value）** ""。：**，。**

： $n$ ， $x_1, x_2, \ldots, x_n$， $p_1, p_2, \ldots, p_n$（ 1），：

$$\mathbb{E}[X] = p_1 \cdot x_1 + p_2 \cdot x_2 + \cdots + p_n \cdot x_n = \sum_{i=1}^{n} p_i \cdot x_i$$

** × ，。**

：， 1 ， 1 。， 50%。 1 ， 1000 ， 500 、500 ：

$$\text{} = 500 \times (+1) + 500 \times (-1) = 0 \quad \Rightarrow \quad \text{} = \frac{0}{1000} = 0$$

——，。：

$$\text{} = \frac{500 \times (+1) + 500 \times (-1)}{1000}$$

 500 = 0.5 × 1000（50%  × 1000 ），：

$$= \frac{0.5 \times 1000 \times (+1) + 0.5 \times 1000 \times (-1)}{1000}$$

 1000，：

$$= 0.5 \times (+1) + 0.5 \times (-1) = 0$$

 $\mathbb{E}[X] = p_1 \cdot x_1 + p_2 \cdot x_2 = 0.5 \times 1 + 0.5 \times (-1) = 0$ 。
:::

，****：

$$\mathbb{E}[R_A] = 0.6 \times (+1) + 0.4 \times (-1) = +0.2$$

$$\mathbb{E}[R_B] = 0.4 \times (+1) + 0.6 \times (-1) = -0.2$$

，A  0.2，B  0.2。。

""；""。RL ：**。**

###  1：

：，A  B  50% 。

****。：（50%/50%），。 4 ：

|           |  |  |             |
| :-----------: | :----: | :--: | :-------------: |
|   A   |   A    |  +1  | 0.5 × 0.6 = 0.3 |
|  A  |   A    |  -1  | 0.5 × 0.4 = 0.2 |
|   B   |   B    |  +1  | 0.5 × 0.4 = 0.2 |
|  B  |   B    |  -1  | 0.5 × 0.6 = 0.3 |

，，：

$$\mathbb{E}[R_{\text{}}] = 0.3 \times (+1) + 0.2 \times (-1) + 0.2 \times (+1) + 0.3 \times (-1) = 0$$

： 4 ，""。

 A， 0.2； B， 0.2。 A、B  0.5 ，：

$$\mathbb{E}[R_{\text{}}] = 0.5 \times 0.2 + 0.5 \times (-0.2) = 0$$

 4 ，：""，""，。

::: details ：""？

，****。



$$C_A = \text{ A}, \quad C_B = \text{ B}$$

$C_A$  $C_B$ ，： A， B。：

$$\mathbb{E}[R_{\text{}}] = P(C_A)\mathbb{E}[R_{\text{}} \mid C_A] + P(C_B)\mathbb{E}[R_{\text{}} \mid C_B]$$

：

$$\text{} = \text{ A } \times \text{A } + \text{ B } \times \text{B }$$

：

$$\mathbb{E}[R_{\text{}} \mid C_A] = 0.6 \times 1 + 0.4 \times (-1) = 0.2$$

$$\mathbb{E}[R_{\text{}} \mid C_B] = 0.4 \times 1 + 0.6 \times (-1) = -0.2$$

：

$$\mathbb{E}[R_{\text{}}] = 0.5 \times 0.2 + 0.5 \times (-0.2) = 0$$

 4 ？****。""，" A "、" A "。

：

$$P(\text{ A }) = P(\text{ A})P(\text{}\mid \text{ A}) = 0.5 \times 0.6 = 0.3$$

$$P(\text{ A }) = P(\text{ A})P(\text{}\mid \text{ A}) = 0.5 \times 0.4 = 0.2$$

，：

$$0.3 \times 1 + 0.2 \times (-1) + 0.2 \times 1 + 0.3 \times (-1)$$

$$= 0.5 \times \big(0.6 \times 1 + 0.4 \times (-1)\big) + 0.5 \times \big(0.4 \times 1 + 0.6 \times (-1)\big)$$

，""；""。
:::

 0。 100 ？：

> **（，Linearity of Expectation）**： $X_1, X_2, \ldots, X_n$，、，：
>
> $$\mathbb{E}\!\left[\sum_{i=1}^{n} X_i\right] = \sum_{i=1}^{n} \mathbb{E}[X_i]$$
>
> ： $\mathbb{E}[X_i]$ 。

，****——。 $\mathbb{E}[R_T] = \sum_{t=1}^{T} \mathbb{E}[R_{a_t}]$ 。， 0， 100  0  0：

$$\mathbb{E}[R_{100}] = \overbrace{0 + 0 + \cdots + 0}^{100} = 0$$

 0 。 A  B 。**：。** ，""，。

###  2： A

 A ， 100  A。 A ：

$$\mathbb{E}[R_{\text{}}] = 0.6 \times (+1) + 0.4 \times (-1) = 0.2$$

，100  = ：

$$\mathbb{E}[R_{100}] = \overbrace{0.2 + 0.2 + \cdots + 0.2}^{100} = 100 \times 0.2 = 20$$

 20，。：****——。，。

###  3：

：**，，。**

： 20  A  B（ 10 ），； 80 。

 20  1 ， 50%  A、50%  B， 0，20  0。

 80  A（）， 0.2：

$$\mathbb{E}[R_{\text{80}}] = 80 \times 0.2 = 16$$

100 ：

$$\mathbb{E}[R_{100}] = 0 + 16 = 16$$

 20—— 4 。

，""： 20 ， 80 。：****。

：""？ RL ， A  60%， B  40%。，。：****。

### 

：**** A 。，，。

：****  **** 。

- 。A  60%，， 60%。
- 。 10 ，。

，A  60%，" 10  6 "。 10  4 ， 10  7 。10 ，。

：

：✅ ，❌ 。

|  |  | 10                  |  |  |
| :--: | :--------: | :---------------------------- | :------: | :--------: |
|  A   |    60%     | ❌ ✅ ❌ ❌ ✅ ❌ ✅ ❌ ✅ ❌ |   4    |    40%     |
|  B   |    40%     | ✅ ❌ ✅ ✅ ❌ ❌ ✅ ❌ ✅ ❌ |   5    |    50%     |

。A  60% ， 10 ；B  40% ， 10 。

，A  B ； 10 ，B 。，。

， B ， 80  B 。""：**，。**

， RL ：**，，。**

::: details ：？

，。

- ****： $p$。。 A  $p_A=0.6$。
- **10 **： $N$。，。
- ****： $\hat p$，"p hat"，。

：

$$\hat p = \frac{N}{10}$$

 A  10  4 ，：

$$N_A = 4,\quad \hat p_A = \frac{4}{10}=0.4$$

 A  $0.6$  $0.4$。：

$$p_A = 0.6$$

，：

$$\hat p_A = 0.4$$

， A，：

$$
X_i =
\begin{cases}
1, & \text{ } i \text{ }\\
0, & \text{ } i \text{ }
\end{cases}
$$

A  $p_A=0.6$，：

$$P(X_i=1)=0.6,\quad P(X_i=0)=0.4$$

：$0.6$ ****， $0.6$。，$X_i$  1  0。

 10 ，：

$$N_A = X_1 + X_2 + \cdots + X_{10}$$

 $X_i$ ， $N_A$ 。 6， 4，。：

$$\hat p_A = \frac{N_A}{10}$$

：

- $p_A=0.6$：。
- $N_A=4$： 10 。
- $\hat p_A=0.4$：。

，$p_A$ ：

$$\mathbb{E}[N_A]=10\times 0.6=6,\quad \mathbb{E}[\hat p_A]=0.6$$

" 10  6 "，" 10 ， 6"。

：**10 ，**。A  10 ， 60%， 0 、1 …… 10 ， 6 。

""****：

$$P(N=k)=\binom{n}{k}p^k(1-p)^{n-k}$$

 $N$  $n$ ，$p$ 。 A  4 ：

$$P(n_A = 4) = \binom{10}{4} \times 0.6^4 \times 0.4^6 \approx 11.1\%$$

，A  60%， 10  4 ， 11.1% 。，。

B  5 ：

$$P(n_B = 5) = \binom{10}{5} \times 0.4^5 \times 0.6^5 \approx 20.1\%$$

，"A  10  4 ，B  10  5 "，，。，：

$$11.1\% \times 20.1\% \approx 2.2\%$$

，。，： A  B，。。

：，，****。 10 。

 A  4 ，：

- $\binom{10}{4}$： 10  4 ，
- $0.6^4$： 4 ， 60%
- $0.4^6$： 6 ， 40%

**？**

$\binom{n}{k}$ " n  k "， $C_n^k$。****， $n$  $k$ ，。

，" 4 "。 10  4 ： 1、2、3、4 ， 1、3、5、7 ，。 $0.6^4 \times 0.4^6$，""， $\binom{10}{4}$。

，$\binom{10}{4}$ ，****。

：****。

： 4  $\{1,2,3,4\}$， 2 。

$$\binom{4}{2} = \frac{4!}{2! \times 2!} = \frac{4 \times 3 \times 2 \times 1}{(2 \times 1)(2 \times 1)} = 6$$

 $C_4^2 = 6$。：

$$\{1,2\}, \{1,3\}, \{1,4\}, \{2,3\}, \{2,4\}, \{3,4\}$$

 $A_4^2$。$A_4^2$ ，， $\{1,2\}$  $\{2,1\}$ ， $A_4^2 = 4 \times 3 = 12$。

：$\binom{10}{4}$， $C_{10}^4$，"10 ， 4 "。 1、3、5、7 ， 2、4、6、8 。

 $A_{10}^4$， 4 ； 1  10 ，""。 $C_{10}^4$。
:::

：** B **。

： 20 ， A  B 。， 80 。

-  A ，， A。
-  B ，， B。
- ， B 。

，：

> **B  A 。**

 $n_A$  A  10 ，$n_B$  B  10 。

？，： A ， B 。。

| A  | B  |   |  |
| :--------: | :--------: | :---------: | :------: |
|     4      |     5      |      B      |        |
|     3      |     6      |      B      |        |
|     5      |     5      | / |        |
|     6      |     4      |      A      |        |

： B  A ，。""，：

$$P(\text{}) \approx 12.8\%$$

： A ， 10 ， 8 ， 1 ， B 。

::: details ：12.8% ？

""，：

$$n_A < n_B$$

 A  B 。

 A ：

|  A ... |  B ... |  |
| :------------: | :------------: | :------: |
|      0       |   1  10    |        |
|      1       |   2  10    |        |
|      2       |   3  10    |        |
|      ...       |      ...       |   ...    |
|      9       |     10       |        |
|     10       |      |        |

：

$$P(\text{})$$

$$=P(A\text{}0)P(B\text{}1)$$

$$+P(A\text{}1)P(B\text{}2)$$

$$+\cdots+P(A\text{}9)P(B\text{}10)$$

$$\approx 12.8\%$$

：

- ？ A  10  B  10 。"A  4  B  5 "， $P(A\text{}4)\times P(B\text{}5)$。
- ？。，A  0  1 ，。
  :::

， 80  B，

$$\mathbb{E}[R_{100} \mid \text{}] = 0 + 80 \times (-0.2) = -16$$

 3 ，""： 87.2% （ 16 ）， 12.8% （ 16 ）。：

$$\mathbb{E}[R_{100}] = 87.2\% \times 16 + 12.8\% \times (-16) \approx 11.9$$

 16  4 。，：**，。**

， A  B （60% vs 40%）。， 52% vs 48%， 50%—— 80 。

****。，，""；，。，。

## 

，" A  B"。：**，。**

""。：""""。

| ****          | ****                      | **** | ****         |
| ----------------- | ----------------------------- | ------------ | ---------------- |
|             |                       |            |    |
|               |             |            |      |
| ε-            |  ε ，1-ε      |  | ε    |
|           |  N              |        | N          |
| UCB               | " + " |  |  |
| Thompson Sampling |                 |      |    |

 UCB（Upper Confidence Bound [^3]） Thompson Sampling [^1] 。""？ **Regret（）** ——。

## Python 

```python
import random

class TwoArmedBandit:
    """： RL """

    def __init__(self, prob_a=0.6, prob_b=0.4):
        self.prob_a = prob_a
        self.prob_b = prob_b

    def step(self, action):
        """，"""
        if action == "A":
            return 1 if random.random() < self.prob_a else -1
        else:
            return 1 if random.random() < self.prob_b else -1
```

""——，。：** MDP**。 CartPole  LLM ——。

##  Python 

，。 `TwoArmedBandit` ，。

###  1：

```python
from random import choice
env = TwoArmedBandit()
total = sum(env.step(choice(["A", "B"])) for _ in range(100))
print(f" 100 : {total}，: {total/100:.2f}")
```

:::output
 100 : -2，: -0.02
:::

 0， 0 ，。

###  2： A

```python
env = TwoArmedBandit()
total = sum(env.step("A") for _ in range(100))
print(f" A 100 : {total}，: {total/100:.2f}")
```

:::output
 A 100 : 18，: 0.18
:::

 20， 18——，。

###  3：

```python
env = TwoArmedBandit()

#  1：——A  B  10 ，
rewards = {"A": [], "B": []}
for arm in ["A", "B"]:
    for _ in range(10):
        rewards[arm].append(env.step(arm))

# ，
avg = {arm: sum(r) / len(r) for arm, r in rewards.items()}
best = max(avg, key=avg.get)  #  avg["A"] = 0.2 > avg["B"] = -0.1 → best = "A"

#  2：—— 80  best
explore_total = sum(sum(r) for r in rewards.values())
exploit_total = sum(env.step(best) for _ in range(80))
total = explore_total + exploit_total

print(f": A ={avg['A']:.2f}, B ={avg['B']:.2f} →  {best}")
print(f" 100 : {total}，: {total/100:.2f}")
```

:::output
 100 : 14，: 0.14
:::

 16（）， 14——" A" 18， 20 。

### 

|      |  |  |                                  |
| -------- | -------- | -------- | ------------------------------------ |
|  | 0        | −2       |  0 ，              |
|  A | 20       | 18       | ， A       |
|  | 16       | 14       |  A  4 ， |

，，。**。**

：

- **。** ，。
- **，。**  A  B  50%，； A  B ， A。

## 

 RL ，。

**1. 。** ，：

- ****（ 0）：，。
- ****（ 20）：，——。
- ****（ ≈ 12）： 20 、 80 。，：****。 60% vs 40% ， 10  12.8% ；，。

**2. 。** （A  +0.2，B  −0.2）， $T$ 。，。

**3.  RL 。** ε-；UCB ；Thompson Sampling 。

""（），"""-"。 RL （CartPole、LLM ），——，""""。 MDP ：[MDP 、](./mdp)

## ：Regret

，。

。（A  60%、B  40%），。****：、，""。 **Regret（）**。

Regret ：，100 ，。，，。 Regret：

> Regret =  − 

（ 100  A， 20 ）：

|      |  | Regret          |
| -------- | ---------- | --------------- |
|  | 0        | 20 − 0 = **20** |
|  A | 20       | 20 − 20 = **0** |
|  | ≈12      | 20 − 12 ≈ **8** |

Regret ，。 100 "" 20 ， 8 （）， A —— A ，。

**RL ， Regret 。**  UCB  Thompson Sampling，—— Regret ， [^4]。。

## 

[^1]: Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. _Biometrika_, 25(3/4), 285-294.

[^2]: Robbins, H. (1952). Some aspects of the sequential design of experiments. _Bulletin of the American Mathematical Society_, 58(5), 527-535.

[^3]: Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. _Machine Learning_, 47(2-3), 235-256.

[^4]: Lai, T. L., & Robbins, H. (1985). Asymptotically efficient adaptive allocation rules. _Advances in Applied Mathematics_, 6(1), 4-22.

[^5]: Sutton, R. S., & Barto, A. G. (2018). _Reinforcement Learning: An Introduction_ (2nd ed.). MIT Press.
