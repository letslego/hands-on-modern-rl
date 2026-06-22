# 6：Actor-Critic——

 4 （Value-Based）： $Q(s,a)$，（：[Q(s,a) ](../chapter03_mdp/value-q)）。，，。 5 （Policy-Based）： $J(\theta)$（：[](../chapter03_mdp/policy-objective)）。、，——，。

：（：[](../chapter05_policy_gradient/pg-improvements)）， $V(s)$（：[](../chapter03_mdp/value-bellman)）。 $V(s)$ ——。 **Critic**。

： Critic ， Actor 。 **Actor-Critic **。

::: tip 
，：

- [ $V(s)$ ](../chapter03_mdp/value-bellman)——Critic ：$V^\pi(s)$ " $s$ ，"
- [ $Q(s,a)$](../chapter03_mdp/value-q)——$Q$  $V$ 
- [DP/MC/TD ](../chapter03_mdp/dp-mc-td)—— Critic 
- [TD Error $\delta = r + \gamma V(s') - V(s)$](../chapter03_mdp/dp-mc-td)——Critic 
- [ $J(\theta)$ ](../chapter03_mdp/policy-objective)——Actor 
- [REINFORCE ](../chapter05_policy_gradient/pg-improvements)—— $V(s)$ 
  :::

## 

|                                             |                                                 |
| ----------------------------------------------- | ------------------------------------------------------- |
| [](./advantage-function)                |  $A(s,a)$ ？ $G_t$ ？       |
| [Critic ](./critic-training)                |  Critic  $V(s)$？DP/MC/TD       |
| [Actor-Critic ](./actor-critic)             | Actor  Critic ？TD Error  $G_t$？     |
| [Actor-Critic ](./ac-frontier)  | AlphaStar、SAC 、Isaac Lab——AC  |
| [：Pendulum ](./pendulum)           | Actor-Critic ？                     |
| [：BipedalWalker ](./bipedalwalker) | Actor-Critic ？                   |

—— Actor  Critic 。[](./advantage-function)
