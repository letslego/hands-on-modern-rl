# E.2.6 

> ****： E.2 ， [E.2.1](./probability-basics)  [E.2.5](./probability-bellman-advanced) 。，。

---

 E.2 ，。。

## 

|             |                                                                                         |                    |
| --------------- | ------------------------------------------------------------------------------------------- | ------------------------------ |
|         | $\pi(a\mid s)$                                                                              |  $s$  $a$  |
|     | $p(s'\mid s,a)$                                                                             |    |
|             | $\mathbb{E}[X]=\sum_x p(x)x$                                                                | 、、   |
|         | $v_\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s]$                                                    |  $s$   |
|             | $\mathrm{Var}(X)=\mathbb{E}[(X-\mathbb{E}[X])^2]$                                           |                |
|     | $\hat{v}(s)=\frac{1}{N}\sum_i G_i$                                                          |              |
|         | $p(\tau\mid\pi)=p(s_0)\prod_t\pi(a_t\mid s_t)p(s_{t+1}\mid s_t,a_t)$                        |          |
| Baseline  | $\mathbb{E}[\nabla\log\pi(a\mid s)b(s)]=0$                                                  |  baseline      |
| GAE             | $`\hat{A}_t^{\mathrm{GAE}}=\sum_k(\gamma\lambda)^k\delta_{t+k}`$      |  TD  MC            |
|       | $\rho=\frac{\pi(a\mid s)}{b(a\mid s)}$                                                      |                      |
| PPO     | $L^{CLIP}=\mathbb{E}[\min(r_t\hat{A}_t,\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\hat{A}_t)]$ |          |

---

## 

，：，，，，。：、，、、。，——、、、，。

---

## 

1. **。** ，。
2. **，。** ，，。
3. **。** ，。

---

## 

1.  $10,4,-2$， $0.2,0.5,0.3$，？
2.  $2,6,10$，？
3.  $0.25$， $0.75$，？
