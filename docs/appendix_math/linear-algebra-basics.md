# E.1.1 ：

> ****：，[](./intro)""。

---

## 、

****。 $r=2$ ， $\gamma=0.9$ 。 RL ，：

|  |              |          |
| ---- | ---------------- | -------------- |
| r    |          | 2、−1、0.5     |
| γ    |          | 0.9、0.99、0.5 |
| α    |            | 0.001、3×10⁻⁴  |
| ε    |  | 0.1、0.2       |

。——****。

****。、：

$$
\mathcal{S}=\{s_1,s_2,s_3\}, \qquad \mathcal{A}=\{\text{left},\text{right}\}.
$$

 $\mathcal{S}$  $\mathcal{A}$ ， $S$  $A$ 。" $s$ "，$s$  $\mathcal{S}$。

，，****。

，：

$$
v:\mathcal{S}\to\mathbb{R}, \qquad s\mapsto v(s).
$$

：

- $v:\mathcal{S}\to\mathbb{R}$ "$v$  $\mathcal{S}$  $\mathbb{R}$ "——，。$\mathbb{R}$ 。
- $s\mapsto v(s)$  $s$  $v(s)$。

： $s$， $v(s)$。 $v(s_1)=3$  $s_1$  $3$。

，，：

$$
\pi:\mathcal{S}\times\mathcal{A}\to[0,1], \qquad (s,a)\mapsto\pi(a\mid s).
$$

：

- $\mathcal{S}\times\mathcal{A}$  $\times$ ——-。
- $[0,1]$  $0$  $1$ ，。
- $(s,a)\mapsto\pi(a\mid s)$ -， $s$  $a$ 。

。$v(s)$ ，。 $p(s'\mid s,a)$ ——"、、"，。

、、""：，-。——， $v(s_i)$ 。，，****。

---

## 

，：

|  |  |
| ---- | ---- |
| s₁   | 3    |
| s₂   | 5    |
| s₃   | 2    |

，：

$$
\boldsymbol{v} =
\begin{bmatrix}
3 \\
5 \\
2
\end{bmatrix}.
$$

****。$\boldsymbol{v}$ ，。，""。

**。**  $1$， $1$：

$$
\boldsymbol{v}_{new} =
\begin{bmatrix}
3 \\
5 \\
2
\end{bmatrix}
+
\begin{bmatrix}
1 \\
1 \\
1
\end{bmatrix}
=
\begin{bmatrix}
4 \\
6 \\
3
\end{bmatrix}.
$$

。——。

**。** 。 $\gamma=0.5$：

$$
\gamma\boldsymbol{v} = 0.5 \times
\begin{bmatrix}
3 \\
5 \\
2
\end{bmatrix}
=
\begin{bmatrix}
1.5 \\
2.5 \\
1.0
\end{bmatrix}.
$$

""。 $v = r + \gamma P v$ ，$\gamma v$ ——。

""，""。 $s_1$  $s_2$  $s_3$，"、"，****。

---

## 

 $s_1, s_2$ ：

-  $s_1$ ， $s_2$。
-  $s_2$ ， $s_1$。

：

$$
P =
\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}.
$$

""，""。 $[0, 1]$ ： $s_1$ ， $s_1$  $0$， $s_2$  $1$。 $[1, 0]$ 。

：

$$
\boldsymbol{v} =
\begin{bmatrix}
3.33 \\
2.67
\end{bmatrix},
$$

$\boldsymbol{v}$  $s_1$ ， $s_2$ 。""。 $\boldsymbol{v}$，：

$$
\text{ }s_1\text{ }
= 0 \times v(s_1) + 1 \times v(s_2) = 2.67,
$$

$$
\text{ }s_2\text{ }
= 1 \times v(s_1) + 0 \times v(s_2) = 3.33.
$$

， $P\boldsymbol{v}$——"，"：

$$
P\boldsymbol{v} =
\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}
\begin{bmatrix}
3.33 \\
2.67
\end{bmatrix}
=
\begin{bmatrix}
2.67 \\
3.33
\end{bmatrix}.
$$

： $s_1$  $s_2$， $2.67$； $s_2$  $s_1$， $3.33$。

### 

，：

|  | → s₁ | → s₂ | → s₃ |
| -------- | ---- | ---- | ---- |
| s₁       | 0.1  | 0.7  | 0.2  |
| s₂       | 0.0  | 0.3  | 0.7  |
| s₃       | 0.5  | 0.5  | 0.0  |

：

$$
P =
\begin{bmatrix}
0.1 & 0.7 & 0.2 \\
0.0 & 0.3 & 0.7 \\
0.5 & 0.5 & 0.0
\end{bmatrix}.
$$

 $n$。 2  3 ， $2\times2$  $3\times3$，： $i$ " $s_i$ ，"。。

---

## 

，。，""。



$$
A=
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix},
\qquad
x=
\begin{bmatrix}
10 \\
20
\end{bmatrix}.
$$



$$
Ax=
\begin{bmatrix}
1\times10+2\times20 \\
3\times10+4\times20
\end{bmatrix}
=
\begin{bmatrix}
50 \\
110
\end{bmatrix}.
$$

——。

### 

， $P$  $v$ ， $i$ ： $s_i$ ，。

：

$$
P=
\begin{bmatrix}
0.7 & 0.3 \\
0.2 & 0.8
\end{bmatrix},
\qquad
v=
\begin{bmatrix}
10 \\
5
\end{bmatrix}.
$$



$$
Pv=
\begin{bmatrix}
0.7\times10+0.3\times5 \\
0.2\times10+0.8\times5
\end{bmatrix}
=
\begin{bmatrix}
8.5 \\
6
\end{bmatrix}.
$$

 $0.7\times10+0.3\times5=8.5$ ： $s_1$ ， $70\%$  $10$ 、$30\%$  $5$ ， $8.5$。

：（ $1$），" × "。 $v = r + \gamma Pv$ " + "。

。， $1$，。。

---

## 

，。

 $n$ ，：

$$
v\in\mathbb{R}^n.
$$

：

$$
P\in\mathbb{R}^{n\times n}.
$$

 $Pv$ ：

$$
(n\times n)(n\times1)=n\times1.
$$

。

$$
v=r+\gamma Pv
$$

，。

### 

 Q 

$$
Q(s,a)=w^\top\phi(s,a),
$$

 $w$  $\phi(s,a)$ 。 $w\in\mathbb{R}^d$， $\phi(s,a)\in\mathbb{R}^d$，。

。：

```
（）128  →  64  → （）2 
```

：

|       |  |      |
| ------- | -------- | -------- |
|  1  | W₁       | 128 × 64 |
|  2  | W₂       | 64 × 2   |

：

$$
h = \sigma(W_1^\top x), \qquad z = W_2^\top h.
$$

$W_1$  $128\times64$， $x$  $128\times1$，$W_1^\top x$  $64\times1$， $h$  $64\times1$。$W_2$  $64\times2$，$W_2^\top h$  $2\times1$， logit。

——，。

::: warning 
，$(A \times B)(B \times C) = A \times C$，。 `RuntimeError: mat1 and mat2 shapes cannot be multiplied`，。
:::

---

## 

：

|  | RL                 |              |
| ---- | ---------------------- | ---------------- |
|  | 、       | r=2，γ=0.9       |
|  | 、 | $S=\{s_1, s_2\}$ |
|  | 、     | v(s)，π(a\|s)    |
|  |  | **v**=[3, 5, 2]ᵀ |
|  |  | P ∈ ℝⁿˣⁿ         |

：，，。—— $v = r + \gamma Pv$。

> ****：[E.1.2 ](./linear-algebra-bellman) —— 、，。
