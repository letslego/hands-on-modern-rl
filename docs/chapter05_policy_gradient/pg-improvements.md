# 5.4 

 CartPole  REINFORCE，：，。：**， $G_t$ ？**

。：。

## 

：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t \right]$$

 $G_t$  $G_t - b(s_t)$， $b(s_t)$ 、：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot \left(G_t - b(s_t)\right) \right]$$

？：

$$\mathbb{E}_{\pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot b(s_t) \right] = \sum_t b(s_t) \cdot \underbrace{\mathbb{E}_{a_t \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t | s_t) \right]}_{= \, 0} = 0$$

：。，$\nabla_\theta \log \pi_\theta(a|s)$ ""，，。

:::details ：$\mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s)] = 0$

：$\sum_a \pi_\theta(a|s) = 1$。 $\theta$ ：

$$\sum_a \nabla_\theta \pi_\theta(a|s) = 0$$

 $\nabla_\theta \log \pi = \frac{\nabla_\theta \pi}{\pi}$， $\nabla_\theta \pi$  $\pi \cdot \nabla_\theta \log \pi$：

$$\sum_a \pi_\theta(a|s) \cdot \nabla_\theta \log \pi_\theta(a|s) = 0$$

 $\mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s)]$。

:::

。****。

## 

，""""。

 CartPole 。， $s$  100 （$V(s) \approx 100$）：

|                     | $G_t$ | $G_t - V(s)$ |  |  |
| ----------------------- | ----- | ------------ | ---------------- | ---------------- |
| ， 150    | 150   | +50          |          |          |
| ， 100  | 100   | 0            |          |            |
| ， 50     | 50    | -50          |          |          |

， $G_t$，——""。，""，""。

""：，。—— $V(s)$，""。

##  $V(s)$

。（ episode ）。，。

 $b(s)$。， $b(s) = V^\pi(s)$ ， [^greensmith2004]。，$V^\pi(s)$ "，，"——，""。

 $G_t - V(s_t)$ ****（Advantage）：

$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$

 REINFORCE ，$G_t$  $Q^\pi(s_t,a_t)$ ，：

$$\hat{A}_t = G_t - V(s_t)$$

- $\hat{A}_t > 0$：，
- $\hat{A}_t < 0$：，
- $\hat{A}_t \approx 0$：，

## 

 $A^\pi(s,a)$ 。""，""。""， $G_t$ ""。

：

- ** 6  Actor-Critic**： Critic  $V(s)$，（ episode ）
- ** 7  PPO**： GAE（Generalized Advantage Estimation）
- ** 9  RLHF**：

## ：

，$V(s)$ （）：

```python
#  V(s)
values = value_net(states_t)
value_loss = nn.MSELoss()(values, returns_t)  #  G_t 

# 
with torch.no_grad():
    values_pred = value_net(states_t)
advantages = returns_t - values_pred  # Â_t = G_t - V(s_t)
policy_loss = -(log_probs * advantages).mean()
```

 $V(s_t)$  $G_t$——，"，"。 $G_t$， $\hat{A}_t = G_t - V(s_t)$。

 **REINFORCE with Value Baseline**。 REINFORCE（ episode ，）， $G_t$  $\hat{A}_t$。

 CartPole  vanilla REINFORCE  REINFORCE + Value Baseline：[：CartPole ](./cartpole-baseline)。

---

[^greensmith2004]: Greensmith, E., Bartlett, P. L., & Baxter, J. (2004). Variance reduction techniques for gradient estimates in reinforcement learning. _Journal of Machine Learning Research_, 5, 1471-1530.
