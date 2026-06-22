# 

。 `code/` ，。

## 

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# ，
pip install -r requirements.txt

# 
pip install -r chapter01_cartpole/requirements.txt
```

## 

|                     |                             |                              |                                                                                 |
| ----------------------- | ------------------------------- | ------------------------------------ | ----------------------------------------------------------------------------------- |
| Ch01 CartPole           | `chapter01_cartpole/`           | `1-ppo_cartpole.py`                  | SB3 PPO  CartPole， SwanLab                                             |
|                         |                                 | `2-pytorch_ppo.py`                   |  PyTorch PPO：Actor-Critic、GAE、clip、                                   |
|                         |                                 | `plot_curves.py`                     |                                                           |
| Ch02 DPO                | `chapter02_dpo/`                | `0-download_model.py`                |  Qwen2.5-0.5B-Instruct                                                          |
|                         |                                 | `1-generate_data.py`                 |                                                             |
|                         |                                 | `2-test_before.py`                   |                                                                           |
|                         |                                 | `3-train_dpo.py`                     | TRL DPO                                                                         |
|                         |                                 | `4-test_after.py`                    |                                                                           |
| Ch03 MDP                | `chapter03_mdp/`                | `two_armed_bandit.py`                |                                                                   |
|                         |                                 | `bellman_equation_verify.py`         |                                                                   |
|                         |                                 | `gridworld_q_learning.py`            | GridWorld Q-Learning                                                    |
| Ch04 DQN                | `chapter04_dqn/`                | `dqn_cartpole.py`                    |  DQN  CartPole                                                          |
|                         |                                 | `double_dqn_cartpole.py`             | DQN  Double DQN                                                               |
|                         |                                 | `dqn_gym_sb3.py`                     | SB3 DQN  CartPole、MountainCar、LunarLander ， SwanLab  |
|                         |                                 | `dqn_atari_sb3.py`                   | SB3 DQN  Atari， wrapper、SwanLab、                             |
|                         |                                 | `export_dqn_curves.py`               |  4  DQN eval CSV                                                  |
|                         |                                 | `dqn_pokemon_red_pyboy.py`           | PyBoy + SB3 DQN                                               |
| Ch05 Policy Gradient    | `chapter05_policy_gradient/`    | `reinforce_cartpole.py`              | REINFORCE  CartPole                                                             |
|                         |                                 | `reinforce_with_baseline.py`         | REINFORCE  baseline                                                           |
|                         |                                 | `actor_critic_cartpole.py`           | Actor-Critic  TD Error                                                            |
| Ch07 PPO                | `chapter07_ppo/`                | `ppo_lunar_lander.py`                | SB3 PPO  LunarLander-v3                                                         |
|                         |                                 | `ppo_from_scratch.py`                |  PyTorch PPO                                                                      |
|                         |                                 | `gae_visualization.py`               | GAE                                                                       |
| Ch08 RLHF               | `chapter08_rlhf/`               | `sft_pipeline.py`                    | SFT                                                                             |
|                         |                                 | `reward_model_training.py`           |                                                                         |
|                         |                                 | `rlhf_ppo_train.py`                  |  PPO-RLHF                                                               |
|                         | `chapter08_rlhf/verl_gsm8k/`    | `run_qwen2_5_0_5b_ppo_single_gpu.sh` | 8.7 veRL + GSM8K                                                    |
| Ch09 Alignment          | `chapter09_alignment/`          | `dpo_hands_on.py`                    | DPO  beta                                                                 |
|                         |                                 | `dpo_math_reward.py`                 |  DPO                                                            |
| Ch09 GRPO/RLVR          | `chapter09_grpo_rlvr/`          | `grpo_mechanism.py`                  | GRPO                                                                        |
|                         |                                 | `grpo_math_reasoning.py`             |  GRPO                                                                 |
|                         |                                 | `rule_based_reward.py`               |                                                                         |
| Ch09 Continuous Control | `chapter09_continuous_control/` | `sac_halfcheetah.py`                 | SAC  HalfCheetah-v4                                                             |
|                         |                                 | `ppo_td3_sac_comparison.py`          | PPO、TD3、SAC                                                                   |
| Ch10 Agentic RL         | `chapter10_agentic_rl/`         | `tool_use_agent.py`                  |                                                                     |
|                         |                                 | `multi_turn_rl.py`                   |                                                                     |
|                         |                                 | `generate_synthetic_data.py`         |                                                                         |
|                         |                                 | `mini_deep_research_grpo.py`         | Mini Deep Research GRPO                                                         |
| Ch11 VLM RL             | `chapter11_vlm_rl/`             | `geometry_counting_dataset.py`       |                                                                       |
|                         |                                 | `multi_modal_reward.py`              |                                                                       |
|                         |                                 | `vlm_grpo_train.py`                  | VLM GRPO                                                                    |
| Ch12 Future Trends      | `chapter12_future_trends/`      | `tree_of_thought.py`                 | Tree of Thought                                                             |
|                         |                                 | `multi_agent_marl.py`                |  GridWorld                                                                  |
| Appendix Pitfalls       | `appendix_common_pitfalls/`     | `debug_reward_hacking.py`            |                                                                         |
|                         |                                 | `debug_training_collapse.py`         |                                                                         |

## 

-  `requirements.txt` 。
- LLM ， GPU 。
-  `output/`、。
-  4  DQN  SwanLab ； `swanlab watch swanlog` 。
