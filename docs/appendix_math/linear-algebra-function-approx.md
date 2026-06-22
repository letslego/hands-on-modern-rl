# E.1.3 、

> ****：[E.1.1 ](./linear-algebra-basics)——。[E.1.2 ](./linear-algebra-bellman)—— $\boldsymbol{v} = \boldsymbol{r} + \gamma P\boldsymbol{v}$。

---

## 

 $\boldsymbol{v} = \boldsymbol{r} + \gamma P\boldsymbol{v}$  $\boldsymbol{v} = (I-\gamma P)^{-1}\boldsymbol{r}$。，：**，**。 $10^{170}$ ， $P$ ， $\boldsymbol{v}$ 。

：，。：

$$
\boxed{\hat{v}(s) = \boldsymbol{w}^\top \boldsymbol{x}(s)}
$$

 $\boldsymbol{x}(s)$  $s$ ，$\boldsymbol{w}$ 。。

。， $\boldsymbol{w}$——？""？****。。

---

## 

：

|                          |  |  |
| ---------------------------- | ------------ | -------- |
|                  | 2            |    |
| GridWorld 10x10              | 100          |    |
|                      | ~10^170      |  |
| （） |          |  |

 3  DP、MC、TD ——、、——：** $v(s)$**。 4  Q-Learning ， (, )  $Q(s,a)$。

，""。：**，**。，。

---

## ：

：，****。

：

$$
\boldsymbol{x}(s) =
\begin{bmatrix}
1 \\
0.5 \\
2
\end{bmatrix}.
$$

" $1$"" $0.5$"" $2$"。：

$$
\boldsymbol{w} =
\begin{bmatrix}
0.2 \\
1.0 \\
-0.1
\end{bmatrix}.
$$

：

$$
\hat{v}(s) = \boldsymbol{w}^\top \boldsymbol{x}(s)
= 0.2 \times 1 + 1.0 \times 0.5 + (-0.1) \times 2 = 0.5.
$$

：**，**。$\boldsymbol{w}$ ，$\boldsymbol{x}(s)$ ，。

 3  $Q(s,a)$ ：

$$
Q(s,a) = \boldsymbol{w}^\top\phi(s,a),
$$

 $\phi(s,a)$  (, ) 。 4  DQN ——，。

### ？

。。

**One-hot **（）：

 $\{s_1, s_2, s_3\}$，：

|  |     |
| ---- | ----------- |
| s_1  | [1, 0, 0]^T |
| s_2  | [0, 1, 0]^T |
| s_3  | [0, 0, 1]^T |

 one-hot ，$\boldsymbol{w}^\top \boldsymbol{x}(s)$  $\boldsymbol{w}$  $i$ ——。，****。 3  Q-Learning ， one-hot 。

****（GridWorld）：

 GridWorld ， $(r, c)$ ：

$$
\boldsymbol{x}(s) = [r/c_{max},\; c/r_{max},\; \text{dist\_to\_goal}]^\top.
$$

。

****：

 RL ，，。 $\boldsymbol{x}(s)$， $\boldsymbol{w}^\top \boldsymbol{x}(s)$。 4  DQN 。

，""：

$$
\text{（one-hot）} \;\to\; \text{（）} \;\to\; \text{（）}
$$

，。（），（ ReLU），。

### 

""。：

$$
\boldsymbol{w}^\top \boldsymbol{x} = \|\boldsymbol{w}\| \|\boldsymbol{x}\| \cos\theta.
$$

 $\theta$ 。：

-  $\boldsymbol{w}$  $\boldsymbol{x}$ （$\cos\theta > 0$），。
- （$\cos\theta < 0$），。
- （$\cos\theta = 0$），。

 RL ，，。$\boldsymbol{w}$ ，$\boldsymbol{x}(s)$ ——，。

---

## ：

， $\hat{v}(s) = \boldsymbol{w}^\top \boldsymbol{x}(s)$ 。 $\boldsymbol{w}$ ？

：。 3  TD  TD Error ：

$$
V(s) \leftarrow V(s) + \alpha\left[r + \gamma V(s') - V(s)\right].
$$

——（），：

$$
\boldsymbol{w} \leftarrow \boldsymbol{w} - \alpha \boldsymbol{g},
$$

 $\boldsymbol{g}$ ，$\alpha$ 。

：** $\boldsymbol{g}$ ，？** ，，；，。""，****——。：

$$
\boxed{\|\boldsymbol{g}\|_2 = \sqrt{\sum_i g_i^2} \quad\longrightarrow\quad \text{ }\|\boldsymbol{g}\|_2 > c\text{， }c}
$$

""，""。。

---

## ：

 **L2 **，：

$$
\|\boldsymbol{x}\|_2 = \sqrt{\sum_i x_i^2}.
$$

 $[3, 4]^\top$， L2 ：

$$
\|[3, 4]^\top\|_2 = \sqrt{3^2 + 4^2} = 5.
$$

—— L2 。

### L1 

 $[3, 4]^\top$  **L1 **：

$$
\|[3, 4]^\top\|_1 = |3| + |4| = 7.
$$

L1 。 L2，L1 （），。

### L1  L2：

|        | L2                 | L1               |
| ---------- | ---------------------- | -------------------- | --- | --- |
|        | \sqrt{\Sigma x_i^2}    | \Sigma               | x_i |     |
|  | （） |            |
|  RL    | 、     | 、 |
|    | `max_grad_norm`        | L1             |

### Frobenius （）

。**Frobenius **：

$$
\|A\|_F = \sqrt{\sum_{i,j} A_{ij}^2}.
$$

：

$$
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \qquad \|A\|_F = \sqrt{1+4+9+16} = \sqrt{30} \approx 5.48.
$$

 RL ，Frobenius 。

---

## ：

，""。，，，。****。

### 

，：

$$
\boldsymbol{g} =
\begin{bmatrix}
12 \\
5 \\
-8 \\
3
\end{bmatrix}.
$$

 L2 ：

$$
\|\boldsymbol{g}\|_2 = \sqrt{144 + 25 + 64 + 9} = \sqrt{242} \approx 15.56.
$$

 $5$， $5/15.56 \approx 0.321$ ：

$$
\boldsymbol{g}_{clipped} = 0.321
\begin{bmatrix}
12 \\
5 \\
-8 \\
3
\end{bmatrix}
\approx
\begin{bmatrix}
3.85 \\
1.61 \\
-2.57 \\
0.96
\end{bmatrix}.
$$

 $5$。**，**——。

 RL  `max_grad_norm` 。 PyTorch ：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

### RL 

RL ，——，。，，。 5  REINFORCE ：。，。

---

## ：

：**，**。

|                     |                           |                   |
| ----------------------- | ----------------------------- | --------------------- |
| （ n < 1000） | ： v(s)     | one-hot +         |
|       | ：v(s) ≈ w^T x(s)     |  +        |
| （、）  | ：v(s) ≈ f\_\theta(s) |  +  |

，。，""，。

---

## ：

""：，。

，：

1. ， $v_{k+1} = r + \gamma Pv_k$ ？ 3  DP ""——，？
2.  3  $\theta \leftarrow \theta + \alpha\nabla_\theta J(\theta)$，？，—— $0.1$ ， $0.1$ 。

：""。、——。

> ****：[E.1.4 、](./linear-algebra-advanced) —— ？？
