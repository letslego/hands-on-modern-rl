# E.3.1 ：、

> ****：，[](./intro)""[E.1.1 ](./linear-algebra-basics)。

---

## 、

——。、""，""。，：，。。

。，。：”，”。，”” $\theta$，”” $J(\theta)$。

：

$$
\theta \longmapsto J(\theta).
$$

 $J(\theta)$ ， $J(\theta)$ 。：**，，？**

，，。

---

## 

""。。。

$$
J(\theta)=-(\theta-0.8)^2+1.
$$

 $\theta$ 。 $\theta=0.2$  right ，$\theta=0.8$  right 。

 $\theta=0.8$ 。，：

| $\theta$ | $J(\theta)$ |
| -------- | ----------- |
| $0.2$    | $0.64$      |
| $0.5$    | $0.91$      |
| $0.8$    | $1.00$      |
| $1.0$    | $0.96$      |

 $0.2$  $0.5$，； $0.8$  $1.0$，。：，。

， $J'(\theta)$（”J prime of theta”，””）：

$$
J'(\theta)=-2(\theta-0.8).
$$

 $\theta=0.2$ ：

$$
J'(0.2)=1.2.
$$

，。 $\theta=1.0$ ：

$$
J'(1.0)=-0.4.
$$

，。

---

## 

，""。，（gradient ascent）：

$$
\theta \leftarrow \theta + \alpha J'(\theta).
$$

 $\theta=0.2$ ， $\alpha=0.1$：

$$
\theta \leftarrow 0.2 + 0.1\times1.2 = 0.32.
$$

：

$$
J'(0.32)=-2(0.32-0.8)=0.96.
$$

：

$$
\theta \leftarrow 0.32 + 0.1\times0.96 = 0.416.
$$

 $0.8$。，。

， $L(\theta)$，，（gradient descent）：

$$
\theta \leftarrow \theta - \alpha L'(\theta).
$$

” + ”，：，。

---

## ：

，。。：

$$
J(\theta_1,\theta_2)=-(\theta_1-1)^2-(\theta_2-2)^2+5.
$$

 $(1,2)$ 。，，（gradient）。 $\nabla$（"nabla"）""：

$$
\nabla J(\theta_1,\theta_2)=
\begin{bmatrix}
-2(\theta_1-1) \\
-2(\theta_2-2)
\end{bmatrix}.
$$

 $(0,0)$，：

$$
\nabla J(0,0)=
\begin{bmatrix}
2 \\
4
\end{bmatrix}.
$$

：$\theta_1$ ，$\theta_2$ ， $\theta_2$ 。""，。 $0.1$，：

$$
\begin{bmatrix}
\theta_1 \\
\theta_2
\end{bmatrix}
\leftarrow
\begin{bmatrix}
0 \\
0
\end{bmatrix}
+0.1
\begin{bmatrix}
2 \\
4
\end{bmatrix}
=
\begin{bmatrix}
0.2 \\
0.4
\end{bmatrix}.
$$

。，，。

---

## ：

，""。——，，，。（chain rule）：，。

：

$$
y = 3\theta, \qquad L=(y-6)^2.
$$

 $\theta=1$， $y=3$，：

$$
L=(3-6)^2=9.
$$

 $\theta$ ，$L$ 。 $y=3\theta$  $L$ ：

$$
L=(3\theta-6)^2.
$$

：

$$
\frac{dL}{d\theta}=2(3\theta-6)\times3.
$$

 $\theta=1$ ：

$$
\frac{dL}{d\theta}=2(3-6)\times3=-18.
$$

 $2(3\theta-6)$  $y$ （ $\frac{dL}{dy}$），$3$  $y$  $\theta$ （ $\frac{dy}{d\theta}$）。：

$$
\frac{dL}{d\theta}=\frac{dL}{dy}\cdot\frac{dy}{d\theta}.
$$

："$\theta$  $y$，$y$  $L$， $\theta$  $L$ 。"（backpropagation）——，，。

---

## ：，

 $d$，。， $\partial$（"partial"）："，"。（partial derivative）。：

$$
J(\theta_1,\theta_2)=-(\theta_1-1)^2-(\theta_2-2)^2+5.
$$

。： $\theta_2$ ，$\theta_1$ ？， $\theta_1$ ？。

 $\theta_1$ ：

$$
\frac{\partial J}{\partial \theta_1}=-2(\theta_1-1).
$$

 $\theta_2$ ：

$$
\frac{\partial J}{\partial \theta_2}=-2(\theta_2-2).
$$

，：

$$
\nabla_\theta J=
\begin{bmatrix}
\frac{\partial J}{\partial \theta_1} \\
\frac{\partial J}{\partial \theta_2}
\end{bmatrix}.
$$

 $(\theta_1,
\theta_2)=(0,0)$ ：

$$
\nabla_\theta J(0,0)=
\begin{bmatrix}
2 \\
4
\end{bmatrix}.
$$

，。， 2 。

---

## ：

，。 $\alpha$ 。



$$
\theta=
\begin{bmatrix}
0 \\
0
\end{bmatrix},
\qquad
\nabla J=
\begin{bmatrix}
2 \\
4
\end{bmatrix},
$$

 $\alpha=0.1$，：

$$
\theta\leftarrow
\begin{bmatrix}
0 \\
0
\end{bmatrix}
+0.1
\begin{bmatrix}
2 \\
4
\end{bmatrix}
=
\begin{bmatrix}
0.2 \\
0.4
\end{bmatrix}.
$$

，——；，——。，，、Adam 、PPO 。

::: warning 
""。。 $10^{-3}$、$3\times10^{-4}$ ，。
:::

---

## 

、。，，（autodiff）。。

：

$$
\theta \to y=3\theta \to L=(y-6)^2.
$$

， $\frac{dL}{dy}$， $\frac{dy}{d\theta}$， $\frac{dL}{d\theta}$。

：，。、，。

---

## 

：

|      |                            | RL               |
| -------- | ------------------------------ | ------------------------ |
|      |  |          |
|      |    |  |
|  |              |        |
|    | ， |  |
|    |          |      |

"、"。——""。

> ****：[E.3.2 ](./calculus-policy-gradient) —— 。
