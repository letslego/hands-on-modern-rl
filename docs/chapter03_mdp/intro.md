#  3 ：MDP、

## 

****

-  MDP 、、、。
-  DP、MC、TD 。
-  $Q(s,a)$  $J(\theta)$ ，。

****

$$
\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle \quad \text{（MDP ：）}
$$

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k} \quad \text{（：）}
$$

$$
V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s], \quad Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a] \quad \text{（：）}
$$

$$
J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{\infty}\gamma^t r_t\right] \quad \text{（：）}
$$

****

 3 。MDP ； $G_t$ ；； $J(\theta)$ 。 DQN、、Actor-Critic  PPO 。

 1  CartPole ——****，，。 2 ****， DPO ，。，。

，：CartPole  reward ？DPO ？，""""——。

：**？** ****——，，。，，****。，。

，——****（Markov Decision Process, MDP）。MDP **、、、** $\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$。，、、Q-Learning、、PPO——，。

， **""**。**** $G_t$ ，****。，****：""""。，、。

，****：

- **Value-based **： $Q(s,a)$，，—— Q-Learning  Q 。
- **Policy-based **： $J(\theta)$，——、Actor-Critic  PPO。

****。 4  Q  $Q(s,a)$， 5  $J(\theta)$， 6  Actor-Critic  7  PPO 。，，****。

## 

|                                     |                                              |
| --------------------------------------- | ---------------------------------------------------- |
| [](./bandit)              | 、               |
| [](./mdp)               |  MDP 、                      |
| [](./value-bellman) | ，               |
| [DP、MC  TD](./dp-mc-td)              | 、       |
| [ Q  Q-Learning](./value-q)         |  GridWorld 、TD 、   |
| [](./policy-objective)      |                      |
| [](./algorithm-taxonomy)    |  On-policy  Off-policy，Online  Offline  |
| [](./reward-design)         | ， |
| [](./panorama)                  | 、             |

## 

，：

-  MDP  $\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$ ；
- ****，、、；
- ——**$Q(s,a)$ **、，**$J(\theta)$ **， Q  PPO。

， MDP 。""，。：[：——](./bandit)。
