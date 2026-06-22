# 5.3 ：Value Baseline 

> ****： `CartPole-v1`  REINFORCE （Value Baseline, VB） REINFORCE， $V(s)$ 、。

> ****：[reinforce_with_baseline.py](https://github.com/letslego/hands-on-modern-rl/blob/main/code/chapter05_policy_gradient/reinforce_with_baseline.py) · [render_cartpole_baseline.py](https://github.com/letslego/hands-on-modern-rl/blob/main/code/chapter05_policy_gradient/render_cartpole_baseline.py) · [reinforce_cartpole.py](https://github.com/letslego/hands-on-modern-rl/blob/main/code/chapter05_policy_gradient/reinforce_cartpole.py) · [requirements.txt](https://github.com/letslego/hands-on-modern-rl/blob/main/code/chapter05_policy_gradient/requirements.txt)

 REINFORCE ：
，
。
，
：
，
。

。
，
，
“”。
 `CartPole-v1`：
，
。
，
：
，；
，。

## 5.3.1 Value Baseline 

。
**baseline** ，
。
Williams  1992  REINFORCE ，
。[^williams1992]
：
，
，
。

，Sutton、McAllester、Singh  Mansour  policy gradient theorem ：

$\nabla_\theta \log \pi_\theta(a \mid s)$
“”。[^sutton1999]
 $G_t$，
 $Q^\pi(s,a)$，
 baseline。
 baseline  $V^\pi(s)$ ，


$$
A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s).
$$

 **Value Baseline**，
 $V(s)$，


$$
A_t = G_t - V(s_t)
$$

 $G_t$ 。
，，
“ baseline”，
“ learned value function  baseline”。
 Critic，
 REINFORCE ：
 episode ，
 Monte Carlo  $G_t$ 。

 baseline 。
，
；
，。
， $b(s)$，
 $V(s)$。
Greensmith、Bartlett  Baxter  baseline  actor-critic，
；
 optimal baseline。[^greensmith2004]
，
PPO、A2C、A3C  $G_t$，
 advantage；
GAE  bias  variance 。

， Value Baseline ：
，
 Actor-Critic  TD 。

## 5.3.2  CartPole  Value Baseline

CartPole  4 ：
、、。
：
。
， `+1` ；
，，episode 。
`CartPole-v1`  `500`，
“”。

 REINFORCE 。
，
。
 REINFORCE “”，
。
，
。
。

Value Baseline “”，
。
 REINFORCE  $t$  $G_t$
；
，
 $V(s_t)$，


$$
A_t = G_t - V(s_t).
$$

，
“”“”。
 $V(s_t)$ ，
“”，
“”。
，
；
，
。

 CartPole ：
，
。

## 5.3.3 

：

```bash
pip install -r code/chapter05_policy_gradient/requirements.txt
```

：

```bash
python code/chapter05_policy_gradient/reinforce_with_baseline.py
```

：

|                        |        |       |                              |
| -------------------------- | -------------- | ------------- | ------------------------------------ |
| Vanilla REINFORCE          | `G_t`          |             |        |
| REINFORCE + Value Baseline | `G_t - V(s_t)` | Value Network |  |

 CartPole 。
：
 $G_t$；
Value Baseline  $V(s_t)$，
 $G_t - V(s_t)$ 。

：

|                                             |                        |
| --------------------------------------------------- | -------------------------- |
| `output/reinforce_baseline_reward_comparison.png`   |      |
| `output/reinforce_baseline_variance_comparison.png` |  |

。

 GIF，
：

```bash
python code/chapter05_policy_gradient/render_cartpole_baseline.py \
  --episodes 500 \
  --seed 0
```

，
 CartPole ，
 GIF  `docs/chapter05_policy_gradient/images/`。

## 5.3.4 

：。

![CartPole  REINFORCE  REINFORCE + Value Baseline 。Value Baseline  500 ，。](./images/reinforce-baseline-cartpole-reward.png)

 episode ，
。
 episode ，
：
，
。
，。

，
 REINFORCE  50  `95.1`。
，
，
。
，
 50  `493.0`，
 CartPole  `500` 。

：
。
，
“”，
。

## 5.3.5 

，
。
 GIF ，
 500 。

**Vanilla REINFORCE：，。**
 `166`。
，
，
。

![Vanilla REINFORCE  CartPole ：，。](./images/cartpole-vanilla-reinforce.gif)

**REINFORCE + Value Baseline：。**
 `355`。
 500 ，
，
。

![REINFORCE + Value Baseline  CartPole ：。](./images/cartpole-reinforce-baseline.gif)

 seed ，
。
 REINFORCE ，
，
。
Value Baseline  $G_t - V(s_t)$ “/”，
。

## 5.3.6 

“”。
：
？

![CartPole  REINFORCE  REINFORCE + Value Baseline 。Value Baseline ，。](./images/reinforce-baseline-cartpole-variance.png)

。
，
 episode ；
，
。

，
 REINFORCE  `100.41`，
Value Baseline  `38.27`。
，
Value Baseline  `38.1%`。
：
，
“”。

## 5.3.7 

 REINFORCE ：

```python
returns_t = torch.FloatTensor(returns)
log_probs = torch.log(action_probs + 1e-8)
loss = -(log_probs * returns_t).mean()
```

 `returns_t`  $G_t$。
，
。
，
“”。

，
：

```python
values = value_net(states_t)
value_loss = nn.MSELoss()(values, returns_t)
```

：
 $s_t$ ，
。
 $G_t$，
：

```python
with torch.no_grad():
    values_pred = value_net(states_t)

advantages = returns_t - values_pred
policy_loss = -(log_probs * advantages).mean()
```

。
 `advantages` ，
，
；
 `advantages` ，
，
。

，，
。
。
“”，
“”。

## 5.3.8 

。
，
；
。
 $V(s_t)$ ，
，
。

，
，
，
。
 $G_t - V(s_t)$ ，
。

“”：
，
 100 ，
 100 ，
 100 ，
。

## 5.3.9 

**：。**
。
CartPole  `+1`。
。

**：。**
，
。
 $V(s)$，
，。
。

**： Actor-Critic。**
 REINFORCE with Value Baseline。
 episode ，
 Monte Carlo  $G_t$ 。
 Actor-Critic  TD ，
。

## 

- CartPole ，、，。
-  REINFORCE  $G_t$ ， episode 。
-  $V(s_t)$， $G_t$  $G_t - V(s_t)$。
- ，，。
-  Critic ； Actor-Critic。

## 

1.  `num_episodes`  `200`，。
2.  `1e-3`  `5e-4`  `2e-3`，。
3.  `advantages.mean()`  `advantages.std()`， 0 。
4.  Value Network  `128`  `32`，。

## 

[^1]: Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. _Machine Learning_, 8(3-4), 229-256. [DOI](https://doi.org/10.1007/BF00992696)

[^2]: Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (1999). Policy gradient methods for reinforcement learning with function approximation. _Advances in Neural Information Processing Systems_, 12.

[^3]: Gymnasium. CartPole-v1 documentation. <https://gymnasium.farama.org/environments/classic_control/cart_pole/>

[^williams1992]: Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. _Machine Learning_, 8, 229-256. DOI: <https://doi.org/10.1007/BF00992696>.

[^sutton1999]: Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (1999). Policy gradient methods for reinforcement learning with function approximation. _Advances in Neural Information Processing Systems_, 12.

[^greensmith2004]: Greensmith, E., Bartlett, P. L., & Baxter, J. (2004). Variance reduction techniques for gradient estimates in reinforcement learning. _Journal of Machine Learning Research_, 5, 1471-1530. <https://jmlr.org/papers/v5/greensmith04a.html>.
