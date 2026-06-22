# E.1.2 

> ****：[E.1.1 ](./linear-algebra-basics)——、。 3 [](../chapter03_mdp/value-bellman)，。

---

## 

 3 ：

$$
V^\pi(s) = \sum_{a} \pi(a|s)\left[R(s,a) + \gamma\sum_{s'} P(s'|s,a)V^\pi(s')\right].
$$

。 $n$ ， $n$ ，。 $n$ ，：

$$
\boxed{\mathbf{v} = \mathbf{r} + \gamma P\mathbf{v}}
$$

 $\mathbf{v}$ ，$\mathbf{r}$ ，$P$ 。：

$$
\boxed{\mathbf{v} = (I - \gamma P)^{-1}\mathbf{r}}
$$

，：，，。

---

## ：

：

-  $s_1$， $2$， $s_2$。
-  $s_2$， $1$， $s_1$。
-  $\gamma = 0.5$。

 $v_1$  $v_2$。：

$$
\mathbf{v} =
\begin{bmatrix}
v_1 \\
v_2
\end{bmatrix},
\qquad
\mathbf{r} =
\begin{bmatrix}
2 \\
1
\end{bmatrix}.
$$

$\mathbf{v}$ ，$\mathbf{r}$ 。； $n$ ， $n$ ，。

---

## ：

 $s_1$  $s_2$， $s_2$  $s_1$。：

$$
P =
\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}.
$$

""，""。：$P\mathbf{v}$  $i$ " $s_i$ ，"。：

$$
P\mathbf{v} =
\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}
\begin{bmatrix}
v_1 \\
v_2
\end{bmatrix}
=
\begin{bmatrix}
v_2 \\
v_1
\end{bmatrix}.
$$

： $s_1$  $s_2$， $v_2$； $s_2$  $s_1$， $v_1$。。

---

## ：

： $\mathbf{v}$、 $\mathbf{r}$、 $P$。：

$$
\begin{aligned}
v_1 &= 2 + 0.5v_2, \\
v_2 &= 1 + 0.5v_1.
\end{aligned}
$$

： $\mathbf{r}$， $\gamma P\mathbf{v}$。：

$$
\mathbf{v} = \mathbf{r} + \gamma P\mathbf{v}.
$$

：

$$
2 + 0.5 \times (0 \cdot v_1 + 1 \cdot v_2) = 2 + 0.5v_2.
$$

：

$$
1 + 0.5 \times (1 \cdot v_1 + 0 \cdot v_2) = 1 + 0.5v_1.
$$

。，。

### 

 $P\mathbf{v}$ 。$P$ （ $1$），" $\times$ "。" =  + "，——****。

|     |                      |   |
| ------- | ------------------------ | ----- |
| **v**   | （）   | n × 1 |
| **r**   |        | n × 1 |
| γP**v** |  | n × 1 |

 $n \times 1$，。 3  DP（）—— $v_{k+1} = r + \gamma P v_k$，。

---

## 

$\mathbf{v} = \mathbf{r} + \gamma P\mathbf{v}$ ，。 $\mathbf{v}$ ：

$$
\mathbf{v} - \gamma P\mathbf{v} = \mathbf{r}.
$$

（$I$ —— $1$， $0$）：

$$
(I - \gamma P)\mathbf{v} = \mathbf{r}.
$$

 $I - \gamma P$ ，：

$$
\mathbf{v} = (I - \gamma P)^{-1}\mathbf{r}.
$$

：

$$
I - \gamma P =
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
- 0.5
\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}
=
\begin{bmatrix}
1 & -0.5 \\
-0.5 & 1
\end{bmatrix}.
$$

：

$$
\begin{bmatrix}
1 & -0.5 \\
-0.5 & 1
\end{bmatrix}
\begin{bmatrix}
v_1 \\
v_2
\end{bmatrix}
=
\begin{bmatrix}
2 \\
1
\end{bmatrix}
\quad\Longrightarrow\quad
v_1 = 3.33,\quad v_2 = 2.67.
$$

### ：

 $(v_1, v_2)$ ：

-  $v_1 - 0.5v_2 = 2$
-  $-0.5v_1 + v_2 = 1$

 $(3.33, 2.67)$。****： $(3.33, 2.67)$  $v_{new} = r + 0.5Pv$， $(3.33, 2.67)$。，。

---

##  $n$ 

 $\mathbf{v} = (I-\gamma P)^{-1}\mathbf{r}$ 。 2  $n$，：

$$
\mathbf{v}_\pi = \mathbf{r}_\pi + \gamma P_\pi \mathbf{v}_\pi.
$$

：

- $\mathbf{v}_\pi \in \mathbb{R}^n$： $\pi$ 
- $\mathbf{r}_\pi \in\mathbb{R}^n$：
- $P_\pi \in\mathbb{R}^{n\times n}$：（$P_\pi[i,j] = \sum_a \pi(a\mid s_i) p(s_j\mid s_i, a)$）

，$P$  $3\times3$，$\mathbf{v}$  $\mathbf{r}$  $3\times1$， $\mathbf{v} = \mathbf{r} + \gamma P\mathbf{v}$ 。：**，**。

### $I - \gamma P$ 

 $I - \gamma P$ 。，。 $0 < \gamma < 1$  $P$ （ $1$），$I - \gamma P$ 。

，$\gamma P$ （） $\rho(\gamma P) \leq \gamma < 1$， $I - \gamma P$  $0$，。E.1.4 。

---

## 

，：，，。，：

1. **。**  $n=10^6$， $I-\gamma P$  $10^6 \times 10^6$， $O(n^3)$，。
2. **。** ，$P$ ，。 3  MC  TD  $P$ 。
3. **。** ，—— $P$。

：

- ****： $v_{k+1} = r + \gamma P v_k$，。
- ****：。
- **TD **：。

 $(I-\gamma P)^{-1}\mathbf{r}$ ，。 3  DP、MC、TD ，" →  → "。

::: warning 
 $\mathbf{v} = (I-\gamma P)^{-1}\mathbf{r}$ ，。，。。
:::

---

## Q 

 $V$。 $Q(s,a)$ ， $V$ 。

### 

 $(s,a)$  Q ：

$$
\mathbf{q} =
\begin{bmatrix}
Q(s_1, a_1) \\
Q(s_1, a_2) \\
\vdots \\
Q(s_2, a_1) \\
\vdots
\end{bmatrix}
\in \mathbb{R}^{|\mathcal{S}||\mathcal{A}|}.
$$

，$\mathbf{r} \in \mathbb{R}^{|\mathcal{S}||\mathcal{A}|}$  $(s,a)$ 。

 $P \in \mathbb{R}^{|\mathcal{S}||\mathcal{A}| \times |\mathcal{S}|}$， $(s,a)$ ， $s'$：

$$
P[(s,a),\, s'] = P(s' \mid s, a).
$$

 $\Pi_\pi \in \mathbb{R}^{|\mathcal{S}| \times |\mathcal{S}||\mathcal{A}|}$  Q "" V ：

$$
\Pi_\pi[\,s,\, (s,a)\,] = \pi(a \mid s).
$$

### V-Q 

$$
\mathbf{v}_\pi = \Pi_\pi \mathbf{q}_\pi
$$

 $i$ ：$\sum_a \pi(a|s_i) Q(s_i, a) = V(s_i)$。 $V(s) = \sum_a \pi(a|s) Q(s,a)$ 。

### Q 

$$
Q^\pi(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s')
$$

：

$$
\mathbf{q}_\pi = \mathbf{r} + \gamma P \mathbf{v}_\pi.
$$

 $\mathbf{v}_\pi = \Pi_\pi \mathbf{q}_\pi$ ， Q ：

$$
\mathbf{q}_\pi = \mathbf{r} + \gamma P \Pi_\pi \mathbf{q}_\pi.
$$

 $\mathbf{q}_\pi = (I - \gamma P \Pi_\pi)^{-1} \mathbf{r}$。

### Q 

$$
Q^*(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) \max_{a'} Q^*(s', a')
$$

：

$$
\mathbf{q}_* = \mathbf{r} + \gamma P \cdot \mathrm{rowmax}(\mathbf{q}_*)
$$

 $\mathrm{rowmax}(\mathbf{q}) \in \mathbb{R}^{|\mathcal{S}|}$  Q 。 max ，，（ Q-Learning）。

###  Q  V 

 $\mathbf{v}_\pi = \Pi_\pi \mathbf{q}_\pi$  $\mathbf{q}_\pi = \mathbf{r} + \gamma P \mathbf{v}_\pi$， $\Pi_\pi$：

$$
\Pi_\pi \mathbf{q}_\pi = \Pi_\pi \mathbf{r} + \gamma \Pi_\pi P \mathbf{v}_\pi
\quad\Longrightarrow\quad
\mathbf{v}_\pi = \underbrace{\Pi_\pi \mathbf{r}}_{\mathbf{r}_\pi} + \gamma \underbrace{\Pi_\pi P}_{P_\pi} \mathbf{v}_\pi.
$$

 $\mathbf{v}_\pi = \mathbf{r}_\pi + \gamma P_\pi \mathbf{v}_\pi$。V  $\mathbf{r}_\pi$  $P_\pi$ ， Q —— $\Pi_\pi$ 。"$Q$  $V$ "。

---

## DP 

 3  DP ：

$$
V(s) \leftarrow \sum_a \pi(a|s)\left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s')\right].
$$

：

$$
\mathbf{v}_{k+1} = \mathbf{r}_\pi + \gamma P_\pi \mathbf{v}_k.
$$

。： $\mathbf{v}_0 = \mathbf{0}$ ， $\mathbf{v} = (I - \gamma P_\pi)^{-1}\mathbf{r}_\pi$。

 $\pi'(s) = \arg\max_a [R(s,a) + \gamma \sum_{s'} P(s'|s,a)V^\pi(s')]$ ： $s$， $\mathbf{r} + \gamma P\mathbf{v}_\pi$ （），。

---

## 

|            | （ 3 ）                                                                  |                                                                               |
| -------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
|  | $V^\pi(s)=\sum_a\pi(a\mid s)\left[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^\pi(s')\right]$ | $\mathbf{v}_\pi = \mathbf{r}_\pi + \gamma P_\pi \mathbf{v}_\pi$           |
|  | $V^*(s)=\max_a\left[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^*(s')\right]$                 | $\mathbf{v}_* = \mathbf{r}_* + \gamma P_* \mathbf{v}_*$（ max）     |
|          | —                                                                                      | $\mathbf{v} = (I - \gamma P)^{-1}\mathbf{r}$                                  |
| V-Q        | $V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a)$                                                | $\mathbf{v}_\pi = \Pi_\pi \mathbf{q}_\pi$                                     |
| Q    | $Q^\pi(s,a)=R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\sum_{a'}\pi(a'\mid s')Q^\pi(s',a')$    | $\mathbf{q}_\pi = \mathbf{r} + \gamma P \Pi_\pi \mathbf{q}_\pi$           |
| Q    | $Q^*(s,a)=R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\max_{a'}Q^*(s',a')$                      | $\mathbf{q}_* = \mathbf{r} + \gamma P \cdot\mathrm{rowmax}(\mathbf{q}_*)$ |
| DP     | $V(s) \leftarrow \sum_a\pi(a\mid s)[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V(s')]$         | $\mathbf{v}_{k+1} = \mathbf{r}_\pi + \gamma P_\pi \mathbf{v}_k$           |

MC  TD ，。

---

## 

 $\mathbf{v} = \mathbf{r} + \gamma P\mathbf{v}$， $\mathbf{v} = (I-\gamma P)^{-1}\mathbf{r}$。，。

：** $v(s)$ **。

|                          |  |  |
| ---------------------------- | ------------ | -------- |
|                  | 2            |        |
| GridWorld 10×10              | 100          |        |
|                      | ~10¹⁷⁰       |        |
| （） |          |        |

，，。， $10^{170}$ ，。

：，****。，——。

> ****：[E.1.3 、](./linear-algebra-function-approx) —— ，。
