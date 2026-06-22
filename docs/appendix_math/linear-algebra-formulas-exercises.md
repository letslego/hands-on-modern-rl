# E.1.5 

> ****： E.1 ， [E.1.1](./linear-algebra-basics)  [E.1.4](./linear-algebra-advanced) 。，。

---

##  3 

、、、、。 3 ，""，。

### ：

 3 ，：

$$
V^\pi(s) = \sum_{a} \pi(a|s)\left[R(s,a) + \gamma\sum_{s'} P(s'|s,a)V^\pi(s')\right].
$$

，$n$ ：

$$
\boldsymbol{v} = \boldsymbol{r} + \gamma P\boldsymbol{v}.
$$

 $\boldsymbol{v} = (I-\gamma P)^{-1}\boldsymbol{r}$。 3  DP  $v_{k+1} = r + \gamma Pv_k$，。

### TD Error：

 3  TD ：

$$
V(s) \leftarrow V(s) + \alpha\left[r + \gamma V(s') - V(s)\right].
$$

，：

$$
\boldsymbol{v} \leftarrow \boldsymbol{v} + \alpha \cdot \boldsymbol{e}_s \cdot \delta,
$$

 $\boldsymbol{e}_s$  $s$  one-hot （ $s$  $1$， $0$），$\delta = r + \gamma V(s') - V(s)$  TD Error。 E.1.2  $\boldsymbol{v} = \boldsymbol{r} + \gamma P\boldsymbol{v}$ ——（），（）。

### Q-Learning ：one-hot + 

 3  Q-Learning  (, ) 。， one-hot ：$Q(s,a) = \boldsymbol{w}^\top \boldsymbol{x}(s,a)$， $\boldsymbol{x}(s,a)$  one-hot 。。

### ：

 3  $J(\theta)$。， $\|\boldsymbol{g}\|_2 \leq c$， $\Delta\theta^\top F\,\Delta\theta \leq \delta$——。

---

## 

 E.1 。，。

|            |                 |           | RL             |
| -------------- | ----------------------- | ------------- | ------------------ |
|            | r=2，γ=0.9              |             | 、       |
|            | **v** ∈ ℝⁿ              |           |    |
|            | P ∈ ℝⁿˣⁿ                |           |  |
|        | (Pv)ᵢ = Σⱼ Pᵢⱼvⱼ        |  +    |    |
|  | **v** = **r** + γP**v** |       |        |
|      | **v** = (I − γP)⁻¹**r** |     |          |
|            | **w**ᵀ**x** = Σᵢ wᵢxᵢ   |           | /  |
| L2         | ‖**x**‖₂ = √Σᵢ xᵢ²      |           | 、   |
|          | A**u** = λ**u**         |           |  |
|    | ρ(γP) ≤ γ < 1           |         |      |
|      | (θ−θₒₗₐ)ᵀF(θ−θₒₗₐ) ≤ δ  |  +  | TRPO     |

，。，""。

---

## 

E.1 ：""""。

|    |                 |      |                              |
| ------ | ------------------------- | ------------------ | ------------------------------------ |
|  | 1000  = 1000  | 、、 | **v** = (I − γP)⁻¹**r**              |
|  |       | 、         | v̂(s) = **w**ᵀ**x **(s)，‖**g**‖₂ ≤ c |
|  | //    | 、   | ρ(γP) ≤ γ < 1，ΔθᵀFΔθ ≤ δ            |

---

## 

1. **。** ， $P$ "，"。
2. **。** $\boldsymbol{v}=(I-\gamma P)^{-1}\boldsymbol{r}$ ，—— 3  DP → MC → TD 。
3. **。** 、、""，。
4. **。** （""），，——。
5. ** L2 。** L1 ，Frobenius ，（）——。

---

## 

### 

1. **。**  $\gamma=0.9$，$v_1=1+0.9v_2$，$v_2=2+0.9v_1$， $v_1,v_2$。

2. **。**  $\boldsymbol{r}$  $P$， $\boldsymbol{v} = (I - \gamma P)^{-1}\boldsymbol{r}$ 。

3. **。**  $[6,8]^\top$， $5$，？

### 

4. **。**  $A = \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix}$ ？，？

5. **。** ， $P = \begin{bmatrix} 0.2 & 0.5 & 0.3 \\ 0.0 & 0.4 & 0.6 \\ 0.7 & 0.3 & 0.0 \end{bmatrix}$， $\boldsymbol{r}=[1, 2, 3]^\top$，$\gamma=0.5$。，。

6. **。**  $s$  $\boldsymbol{x}(s)=[0.3, -0.5, 0.8]^\top$， $\boldsymbol{w}=[2, -1, 3]^\top$。 $Q(s,a) = \boldsymbol{w}^\top\boldsymbol{x}(s)$。 $Q(s,a)$  $5$，？

### 

7. **。**  $\gamma=0.95$， $\|\boldsymbol{e}_0\|=100$。 $0.01$ ？（：$\|\boldsymbol{e}_k\| \leq \gamma^k \|\boldsymbol{e}_0\|$）

8. **。** $F = \begin{bmatrix} 2 & 0 \\ 0 & 8 \end{bmatrix}$，$\Delta\theta = [0.3, 0.1]^\top$，$\delta = 0.5$。？ $\Delta\theta = [0.3, 0.2]^\top$ ？""。
