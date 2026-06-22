#  9 ：（DPO / GRPO / RLVR）

 [RLHF ](../chapter08_rlhf/standard-rlhf-pipeline)。，：（Actor、Ref、Critic、Reward Model）、 on-policy 、Reward Model 、 reward hacking 。

，：**，""？**

—— 2023-2025 。，、、。，""—— LLM 。

## ，

""，：

** Reward Model。** RLHF ，Reward Model （" A  B "）。 DPO（Direct Preference Optimization，2023）：， RM ""。，， RLHF 。""""。

** Critic。** PPO  Critic （[ 6 ](../chapter06_actor_critic/advantage-function)：$A_t = \sum \lambda^l \delta_{t+l}$）， Critic  Actor ，。GRPO（Group Relative Policy Optimization，2025）： Critic？ prompt ，， Critic 。""。

**。**  RLHF  DPO  chosen/rejected ，（）""。，，——""。RLVR（Reinforcement Learning with Verifiable Rewards，2025），。

。：**RL  PPO ，**。RLHF（[](../chapter08_rlhf/ppo-rlhf-loop)），DPO ，GRPO ，RLVR 。""，。

## ：

""——、、。 2024-2025 ：** RL ，，**。

DeepSeek-R1 ：， RLVR ，""（Chain-of-Thought）——、、，。 RL ""，""。

 LLM ：

|                          |     |                        |
| ---------------------------- | ----------- | ------------------------------ |
| ****（、） | DPO / RLHF  | （chosen vs rejected） |
| ****（） | GRPO / RLVR | （）       |

。（DPO ），（GRPO / RLVR / DAPO）。 RL ，。

## ：

"DPO  RLHF""GRPO  PPO "。：

- **RLHF（[](../chapter08_rlhf/intro)）** ""——，。。
- **DPO**  RLHF ""——、，。，DPO 。
- **GRPO / RLVR** ——，。 RLHF ，： RLHF ， GRPO/RLVR 。

。DeepSeek-R1  SFT+RLHF ， RLVR 。Qwen3 ， DPO  RLHF（ DPO ）， GRPO。，。

## 

|                                                                                   |                                                                       |                                 |
| ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | --------------------------------------------- |
| [DPO 、](./dpo-theory-and-family)                                       | DPO  RL ？DPO/KTO/SimPO ？          |  DPO ，   |
| [：DPO ](./dpo-hands-on)                                                  | DPO ？Reward Accuracy、Margin、β ？       |  DPO ，             |
| [：GRPO ](../chapter09_grpo_rlvr/grpo-practice-and-mechanism)       | GRPO  Critic？ Critic ？                |  GRPO ， PPO      |
| [DeepSeek-R1  DAPO](../chapter09_grpo_rlvr/deepseek-dapo)                           |  RL  SFT ？ SimpleRL ？DAPO  GRPO？ |  R1-Zero、SimpleRL  DAPO  |
| [RLVR：](../chapter09_grpo_rlvr/rlvr)                                       |  RM ？1-Shot RLVR ？                            |  RLVR               |
| [： API  GRPO](../chapter09_grpo_rlvr/financial-tool-calling-grpo)    |  API ？ GRPO ？     |  tool-calling verifier        |
| [RL Scaling ](../chapter12_future_trends/rl-scaling-outlook)                | Online vs Offline ？RL Scaling ？                           |  RL                 |
| [——](../chapter09_grpo_rlvr/on-policy-distillation) |  RL ？Teacher  log-prob  reward？           |  RL                 |
| [](./industrial-post-training)                                    |  SFT、RLHF、DPO、RLVR、Agentic RL ？                |             |

？ DPO ——[DPO 、](./dpo-theory-and-family)。
