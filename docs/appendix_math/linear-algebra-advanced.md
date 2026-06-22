# E.1.4 、

> ****：[E.1.2 ](./linear-algebra-bellman)—— $\boldsymbol{v} = \boldsymbol{r} + \gamma P\boldsymbol{v}$。[E.1.3 ](./linear-algebra-function-approx)—— L2 。

---

## 

 $\boldsymbol{v} = \boldsymbol{r} + \gamma P\boldsymbol{v}$，。：。 $v_{k+1} = r + \gamma P v_k$  SGD  $\boldsymbol{w} \leftarrow \boldsymbol{w} - \alpha \boldsymbol{g}$，。

：

1.  $v_{k+1} = r + \gamma P v_k$，**，？**
2. ，，****——。

，、。：

****

$$
\boxed{\rho(\gamma P) \leq \gamma < 1 \quad\Longrightarrow\quad \|\boldsymbol{e}_{k}\| \leq \gamma^k \|\boldsymbol{e}_0\| \to 0}
$$

****

$$
\boxed{\|\boldsymbol{g}_{clipped}\|_2 = \min\!\left(\|\boldsymbol{g}\|_2,\; c\right)}
$$

****

$$
\boxed{\Delta\theta^\top F\,\Delta\theta \leq \delta}
$$

。

---

## ：

### ：

：

$$
x_{k+1} = 0.5\,x_k.
$$

 $x_0=8$， $8 \to 4 \to 2 \to 1 \to 0.5 \to \cdots$。 $0.5$（ $1$）， $0$。

。 $0.5$  $0.3$：

$$
A = \begin{bmatrix} 0.5 & 0 \\ 0 & 0.3 \end{bmatrix}.
$$

 $A^k$  $0.5^k$， $0.3^k$。 $k$ ， $0$。

：** $0.5$  $0.3$ **。$0.5$  $0.3$ ****。

### 

， $\boldsymbol{u}$  $\lambda$，：

$$
A\boldsymbol{u} = \lambda \boldsymbol{u},
$$

 $\boldsymbol{u}$  $A$ ****，$\lambda$ ****。： $A$  $\boldsymbol{u}$ ，， $\lambda$ 。

### 

，， $\gamma P$  $1$。

 $2\times2$ ：

$$
A = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}.
$$

 $\det(A - \lambda I) = 0$：

$$
\det\begin{bmatrix} 4-\lambda & 1 \\ 2 & 3-\lambda \end{bmatrix} = 0.
$$

：

$$
(4-\lambda)(3-\lambda) - 2 = 0 \quad\Longrightarrow\quad \lambda^2 - 7\lambda + 10 = 0.
$$

：

$$
\lambda = \frac{7 \pm \sqrt{49-40}}{2} = \frac{7 \pm 3}{2}.
$$

 $\lambda_1 = 5$，$\lambda_2 = 2$。：， $5$ ， $2$ 。 $A$ ，$\lambda_1 = 5$ 。

：** $1$，（）； $1$，（）。**

### 

 $0.5$ ， $1$ 。。

$$
\boldsymbol{v}_{k+1} = \boldsymbol{r} + \gamma P\boldsymbol{v}_k.
$$

 $\boldsymbol{v}^*$， $\boldsymbol{v}^* = \boldsymbol{r} + \gamma P\boldsymbol{v}^*$。，：

$$
\boldsymbol{v}_{k+1} - \boldsymbol{v}^* = \gamma P(\boldsymbol{v}_k - \boldsymbol{v}^*).
$$

 $\boldsymbol{e}_k = \boldsymbol{v}_k - \boldsymbol{v}^*$，：

$$
\boldsymbol{e}_{k+1} = \gamma P\,\boldsymbol{e}_k.
$$

 $x_{k+1} = 0.5\,x_k$ ， $\gamma P$。

$P$ ， $1$，（） $\rho(P) \leq 1$。：

$$
\rho(\gamma P) \leq \gamma < 1.
$$

 $\gamma P$  $1$。——****。

 3  DP ""。$\gamma < 1$ ——。

### $\gamma$ 

$\gamma$ ，。

| γ    | ρ(γP)  |  |                    |
| ---- | ------ | -------- | ---------------------- |
| 0.1  | ≤ 0.1  |      | ， |
| 0.5  | ≤ 0.5  |      |          |
| 0.9  | ≤ 0.9  |      | ， |
| 0.99 | ≤ 0.99 |      |        |

： $\|\boldsymbol{e}_0\|=10$，$k$ ：

|  k | γ=0.5  | γ=0.9  | γ=0.99  |
| ---------- | ---------- | ---------- | ----------- |
| 1          | 5          | 9          | 9.9         |
| 5          | 0.31       | 5.9        | 9.51        |
| 10         | 0.01       | 3.49       | 9.04        |
| 50         | ≈ 0        | 0.005      | 6.05        |
| 100        | ≈ 0        | ≈ 0        | 3.66        |

$\gamma$  $1$，，""。 RL "-"。

### ：

""。——****。 $\mathcal{T}\boldsymbol{v} = \boldsymbol{r} + \gamma P\boldsymbol{v}$ ：

$$
\|\mathcal{T}\boldsymbol{u} - \mathcal{T}\boldsymbol{v}\|_\infty \leq \gamma \|\boldsymbol{u}-\boldsymbol{v}\|_\infty.
$$

：**， $\gamma$ **。 $\gamma < 1$，。—— $\boldsymbol{v}^*$。

<details>
<summary>：Banach </summary>

（Banach ）：，，。$\boldsymbol{v}^*$ 。

，：**$\gamma < 1$ **。 $\gamma = 1$（），。

</details>

---

，，。E.1.3  L2 ，。：****， L2 。，""""。

---

## ：

### 

L2 ：

$$
\|\boldsymbol{x}\|_2^2 = \boldsymbol{x}^\top \boldsymbol{x}.
$$

 $\boldsymbol{x} = [3, 4]^\top$， $\boldsymbol{x}^\top\boldsymbol{x} = 9 + 16 = 25$。L2 —— $1$  $1$ ，""。

，。：

** A**：$\Delta\theta = [1, 0]^\top$

。" left  logit"， $1$  $0.5$  $0.73$。

** B**：$\Delta\theta = [0, 1]^\top$

。""， $1$ 。

 L2  $1$，。 L2  $\|\Delta\theta\|_2 \leq \delta$ 。

### ：

， $F$ ****， L2 ：

$$
\|\boldsymbol{x}\|_F^2 = \boldsymbol{x}^\top F\,\boldsymbol{x}.
$$

：

$$
F = \begin{bmatrix} 1 & 0 \\ 0 & 4 \end{bmatrix}, \qquad \boldsymbol{x} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}.
$$

：

$$
\boldsymbol{x}^\top F\,\boldsymbol{x}
= \begin{bmatrix} 1 & 1 \end{bmatrix}
\begin{bmatrix} 1 & 0 \\ 0 & 4 \end{bmatrix}
\begin{bmatrix} 1 \\ 1 \end{bmatrix}
= 1 \times 1 + 4 \times 1 = 5.
$$

 $4$ ，。 $F$ ，。

### 

TRPO ：

$$
(\theta - \theta_{old})^\top F(\theta - \theta_{old}) \leq \delta.
$$

 $F$  **Fisher **。：**，**。，$F$ ；，$F$ 。

Fisher  $F$  $(i,j)$ ：

$$
F_{ij} = \mathbb{E}_\pi\left[\frac{\partial \log \pi_\theta(a\mid s)}{\partial \theta_i}\frac{\partial \log \pi_\theta(a\mid s)}{\partial \theta_j}\right].
$$

$F$ ——，。

###  TRPO  PPO：

TRPO ，（ Fisher ）。PPO  RL ，：

-  $F$， $r_t(\theta) = \pi_\theta(a_t\mid s_t)/\pi_{old}(a_t\mid s_t)$ 。
- ， `clip(r_t, 1-\epsilon, 1+\epsilon)` 。

：

$$
\|\Delta\theta\|_2^2 = \Delta\theta^\top \Delta\theta \quad\longrightarrow\quad \Delta\theta^\top F\,\Delta\theta \leq \delta \quad\longrightarrow\quad \text{clip}(r_t,\; 1-\epsilon,\; 1+\epsilon)
$$

——****——。

::: warning 
TRPO ，。 $\|\Delta\theta\|_2 = 0.1$，。
:::

---

## 

，——：

|    |          |               |                   |
| ------ | ------------ | ----------------- | --------------------- |
|    |    | 、    | ρ(γP) ≤ γ < 1         |
|    |  | L2 、 | ‖**g**\_clipped‖₂ ≤ c |
|  |    | 、  | ΔθᵀFΔθ ≤ δ            |

：****，****，****。，。

---

## E.1 ：

 E.1 。：** $V(s) = R(s) + \gamma\sum P V(s')$ ？** ，：

```
3
  │
  ▼ ：，
E.1.1 + E.1.2  、、
  │  → v = r + γPv  
  │  → v = (I-γP)⁻¹r  
  │  ，
  ▼ ：，
E.1.3  、、
  │  → ŵ(s) = wᵀx(s)  
  │  → ∥g∥₂ ≤ c  
  │  ，？
  ▼ ：
E.1.4  、、
     → ： +  + 
```

 RL ：

|                      |          | RL              |  |
| ------------------------ | ---------------- | ------------------- | ---------- |
| ？     | 、       |         | E.1.1      |
| ？     |        |   | E.1.2      |
| ？ | 、       |   | E.1.3      |
| ？       | 、 |             | E.1.4      |
| ？     | 、 | TRPO/PPO  | E.1.4      |

> ****：[E.1.5 ](./linear-algebra-formulas-exercises) ——  3 ， RL 。
