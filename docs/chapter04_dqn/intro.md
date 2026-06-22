#  4 ： Q 

## 

****

-  Q-Learning ，。
-  Q  $Q(s,a)$， Q 。
-  DQN ，。
-  LunarLander ， DQN 、。

****

$$
Q(s,a;\theta)\approx Q^*(s,a)
\quad \text{（： Q ）}
$$

$$
y = r+\gamma\max_{a'}Q(s',a';\theta^-)
\quad \text{（DQN  TD ：）}
$$

$$
\mathcal{L}(\theta)
=
\mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}
\left[
\left(
r+\gamma\max_{a'}Q(s',a';\theta^-)
-Q(s,a;\theta)
\right)^2
\right]
\quad \text{（DQN ： TD Error）}
$$

****

 4  3  **$Q(s,a)$ **。 DQN ：-， $\theta$ 。 Q-Learning  TD ： $r$ 。 TD ， $\mathcal{D}$ ，$\theta^-$ 。，， 3 。

 3 ， Q-Learning ： $s$  $a$， $r$， $s'$， TD  $Q(s,a)$。 GridWorld ，。16 、4 ， 64 ；，；，。

，：、。LunarLander  8 ，、、，；Atari ，、、。“TD Target ”，“、”。

，Q-Learning ，“”。DQN ，：， Q 。，， $\theta$；，。

。， DQN “”；，。， Q ： **Q ** ， **** 。

，：** Q-Learning 、。**  Q ， DQN ， LunarLander ，、Q 、， Double DQN、Dueling DQN、Rainbow， LunarLander 。

## 

|                                                       |                                                            |
| --------------------------------------------------------- | ------------------------------------------------------------------ |
| [ Q ](./from-q-to-dqn)                  | Q ？？                         |
| [ Q ](./dqn-components)                 | Q 、、？？ |
| [：LunarLander ](./lunar-lander)                  | 、 DQN ？                        |
| [ Q ](./dqn-family)                           | Double、Dueling、Rainbow ？                      |
| [： LunarLander ](./visual-game-projects) |  Q ？                        |

## 

，：

-  Q-Learning ；
-  Q  $Q(s,a)$；
-  Q  TD ；
- ；
-  DQN ： batch 、loss 、，；
- ， ε-greedy 、；
-  Q ，。

：[ Q ](./from-q-to-dqn)。
