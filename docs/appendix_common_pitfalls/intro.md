#  A：

 DQN、Actor-Critic、PPO， RLHF、GRPO  Agentic RL 。，：

> ，，，、、，？

。“”，：，，，。， batch ；RL ，，。

“”，。：，。

，：

1. ，？
2. Reward、Loss、KL、Entropy、Value Loss、、？
3.  RL ，，？

## 

。，， LLM RL ；： RL “ →  →  → ”。

```mermaid
flowchart LR
    P["<br/>Policy"] --> R[" / <br/>Rollout / Generation"]
    R --> E[" / <br/>Environment / Tools"]
    E --> S[" / <br/>Trajectory / Completion"]
    S --> W["<br/>Reward Signal"]
    W --> T["<br/>Returns / Advantages / Weights"]
    T --> U["<br/>Policy Update"]
    U --> P

    W -.-> V["<br/>Critic / Value Head<br/>PPO "]
    V -.-> T
    P -.-> K["<br/>Reference Policy<br/> KL "]
    K -.-> T
    P -.-> B[" / <br/>Eval & Audit"]
    B -.->|""| W
```

，“reward ”。。

。

****。、、reward model、verifier，。

****“、”。 PPO / Actor-Critic ， return、value target  advantage； advantage “”。 Critic ，advantage ，；。 GRPO / RLVR ， Critic， prompt ， reward  advantage-like 。TRL  GRPO 、 advantage、 KL、 loss， advantage  reward ， Critic [^trlgrpo]。

****。 checkpoint、 reward hacking、，。“”， reward signal。

，“”，“ Agentic RL ”。PPO-RLHF  Critic + KL ；GRPO/RLVR “ + reward/verifier + ”；Agentic RL ，、 verifier /。，；，； Critic ，PPO  advantage ； KL ，；，。

::: tip 
RL “”，“”。
:::

## 

，。、 batch size、 KL ，。，，。

。，：、、、，。

### 

，、、、checkpoint、。， seed [^drltm]。，“”“”。

### 

 reward ，。，：

- ****： reward、policy loss、KL、entropy ，。
- ****： held-out benchmark、、，。
- ****： agent ，。

， RLHF ， reward ，，“”，。

### 

，。： reward 、 reward ， checkpoint 。

，reward hacking ：、、，。 Agentic RL ，，。

### 

，：、 batch、 prompt、。，：

- ；
- reward ；
- ；
-  PPO/Actor-Critic， rollout；
-  GRPO/RLVR， prompt  reward 。

 RL 。 `done` mask 、、padding token  loss、 temperature ，，。，，。

## 

。，。

。，，，。，。

。、、、，。，。

。，，。，，。

。PPO ； Critic ；GRPO/RLVR  reward ；Agentic RL 。

。，。

## ：

 bug，。

 CartPole  action  0/1，； MuJoCo  `[-1, 1]`， tanh； padding token  mask ，“”； agent ，。

：，，。

### 

，：

```python
def sanity_check_env(env, policy):
    obs, info = env.reset(seed=0)
    assert obs is not None

    action = policy.sample(obs)
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert next_obs is not None
    assert isinstance(float(reward), float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    return {
        "reward": reward,
        "done": terminated or truncated,
        "info_keys": list(info.keys()),
    }
```

： 100 ， reward 。“” 100 。，，。

::: warning 
，， reward sign 、terminal state 、action scale 、observation normalization 、 chosen/rejected 。
:::

## ：

RL “”。， prompt、 reward、 KL 、 checkpoint，。

 Agentic RL 。， benchmark、 judge、。

：

|             |                  |    |
| --------------- | -------------------- | -------------- |
| smoke set       |      |            |
| dev set         | 、 reward    | ， |
| public test     |              |            |
| private test    |              |        |
| human audit set |  reward  judge |        |

：temperature、top_p、max_tokens、prompt 、、、pass@1/pass@k 。ALE ：、， RL [^ale]。

## ： reward 

 reward “”， transition、。：，。

， reward 。， reward 。 99.9%  reward  0，。

###  reward 

 reward histogram，。

|          |          |                             |
| ---------------- | ---------------- | ------------------------------- |
|  0         |        | 、、  |
|  1         |        | 、      |
|          |  | reward clipping / normalization |
|        |      |                 |
|  | proxy      |  reward       |

 PPO ，reward  advantage。reward ，advantage ，。 reward normalization、advantage normalization、gradient clipping，[^implementation][^whatmatters]。

## ：

“”，，。AI safety  specification gaming：，[^concrete][^weng]。

：reward model ，、、。Reward ，。Reward model overoptimization ，，[^overopt]。

### 

：

1. **Reward **：。
2. ****：、、、、。
3. ****：、、。

```python
def audit_reward_hacking(samples):
    suspicious = []
    for item in samples:
        if item["reward"] > 0.9 and item["human_score"] < 0.4:
            suspicious.append(("reward-human mismatch", item["id"]))
        if item["response_len"] > item["baseline_len"] * 2:
            suspicious.append(("length inflation", item["id"]))
        if item["repeat_ratio"] > 0.2:
            suspicious.append(("repetition", item["id"]))
    return suspicious
```

，。 reward ：、、、、、。RewardBench ，reward model ，[^rewardbench]。

## ：PPO 

PPO “”。TRPO  KL ，PPO  clipped surrogate objective [^trpo][^ppo][^spinningup]。 clip 。

、PPO epochs 、batch 、advantage ，。

### 

|           |                     |          |
| ------------- | ------------------------- | ------------------ |
| KL divergence | / |        |
| clip fraction |  clip         | PPO  |
| entropy       |         |  |

 reward ， KL、clip fraction、entropy 。Reward 。

```python
def ppo_guardrail(metrics):
    if metrics["kl"] > metrics["target_kl"] * 2:
        return "stop update: KL too high"
    if metrics["clip_fraction"] > 0.4:
        return "reduce lr or PPO epochs"
    if metrics["entropy"] < metrics["entropy_floor"]:
        return "increase exploration or KL constraint"
    return "continue"
```

 RLHF ， reference model  KL。InstructGPT  KL penalty， RL  SFT [^instructgpt]。

## Critic：PPO / Actor-Critic 

 Critic  value head ， Actor-Critic、PPO、 PPO-RLHF 。GRPO/RLVR  Critic ， reward、KL  loss 。

 Actor-Critic ，Critic 。， policy loss。 Critic ，advantage ；advantage ，Actor 。

### Critic 

|                                |                                 |
| ---------------------------------- | ----------------------------------- |
| value loss                 | Critic                  |
| explained variance < 0             |                       |
| policy reward                  | Actor  advantage      |
| value prediction  return | reward scale  value target  |

： reward scale、 return、/ critic learning rate、 critic 、 bootstrap target、 terminal mask。

： rollout， actor， critic， return 。， critic。

## ：

。

 entropy ：，。 entropy ：，reward 。

|                  |                            |                                  |
| -------------------- | ---------------------------------- | ------------------------------------ |
| entropy      | 、KL 、        |  entropy bonus、 lr、 KL |
| entropy      | 、、 | 、、 advantage |
|      |                  |  reward  curriculum            |
|  reward  |  reward hacking                |  reward                |

，“”，、、、/。 token entropy ，。

## ：on-policy 

PPO  on-policy ：“”。 old logprob，。

 rollout worker  learner ， buffer ，：loss ，，，clip fraction 。

：

1.  rollout  policy version？
2.  old logprob ？
3. rollout ，？

Agentic RL ，，，。，。

## ：NaN 

NaN 。 grad norm 、logprob 、reward 、value loss 、。

|            |                  |                        |
| -------------- | -------------------- | -------------------------- |
| grad norm  | p95 / max grad norm  | gradient clipping、 lr |
| logprob    |  0  log  | clamp、 mask           |
| fp16       | loss scale、NaN step | bf16、 loss scaling    |
| reward     | reward max/min       | clipping、normalization    |
| value      | value target     | return normalization       |

 loss  NaN 。，。

## ：

RLHF/PPO  SFT ， actor、critic、reference model、reward model， rollout、logprob、value、advantage 。

：

|        |               |                           |
| ---------- | ------------------------- | --------------------------------- |
|    |               | 、、 rollout/training |
|  | Adam /        | ZeRO、FSDP、8-bit optimizer       |
|        |         | LoRA、                    |
|        | batch  seq_len  | checkpointing、           |

ZeRO 、[^zero][^deepspeedzero]；FSDP  all-gather [^fsdp]；LoRA ，[^lora]。“”， RL 。

 OOM。、GPU 、rollout worker 、reward model ，、，。

## RLHF  Agentic RL 

 agent  RL，。

|        |        |                            |
| ---------- | -------------- | ------------------------------ |
| RLHF       |        | ，   |
| RLHF       |        |  reward ， |
| RLHF       | judge      | LLM judge          |
| RLVR/GRPO  |        |  |
| Agentic RL |        |            |
| Agentic RL |      | ，   |
| Agentic RL |  |        |

，Agentic RL ，、、、、。RLHF  reward model，、、、、。

## 

：reward ，benchmark ，。

“”。：

1. ****：benchmark  temperature、max_tokens ？
2. ****： reward 、、？
3. ****：reward 、、？
4. **KL  entropy**：，？
5. ****：，。
6. ****： reward ， reward 。

：reward ，KL ，clip fraction  0.5。

：

1.  checkpoint。
2.  learning rate。
3.  PPO epochs。
4.  target KL early stop。
5.  advantage normalization  reward scale。

。“reward ”；“？”

## 、、

### 

|            |                                   |
| ---------------- | ------------------------------------- |
|          | reset/step/done/reward ？ |
|      |  reward ？          |
|        | ？        |
| reward histogram | reward  0、 1 ？    |
| eval config      | ？              |
|          | 、batch、seq_len ？ |

### 

|                    |                                        |
| ---------------------- | ------------------------------------------ |
| KL                 | ， lr  KL                  |
| clip fraction  |  PPO epochs                  |
| entropy        |  reward hacking                  |
| value loss         |  Critic                  |
| reward  eval       |  reward                      |
| response length    |                                |
| OOM          |  micro batch / seq_len， ZeRO/FSDP |

### 

|                |                     |
| -------------------- | ------------------------- |
| best eval checkpoint |         |
| last checkpoint      |       |
|  checkpoint      |           |
| reward audit     |  reward hacking |
|  seed          |               |
|            |           |

## 

“”，。

；； Critic ；；。

，“”。：

> ？？？

 RL “”。

## 

[^ppo]: Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017.

[^spinningup]: OpenAI Spinning Up, [Proximal Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/ppo.html).

[^trpo]: Schulman et al., [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477), 2015.

[^instructgpt]: Ouyang et al., [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155), 2022.

[^trlgrpo]: Hugging Face TRL, [GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer).

[^drltm]: Henderson et al., [Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560), 2018.

[^implementation]: Engstrom et al., [Implementation Matters in Deep RL: A Case Study on PPO and TRPO](https://openreview.net/forum?id=r1etN1rtPB), 2020.

[^whatmatters]: Andrychowicz et al., [What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990), 2020.

[^ale]: Machado et al., [Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents](https://arxiv.org/abs/1709.06009), 2018.

[^concrete]: Amodei et al., [Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565), 2016.

[^weng]: Lilian Weng, [Reward Hacking in Reinforcement Learning](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/), 2024.

[^overopt]: Gao et al., [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760), 2022.

[^rewardbench]: Lambert et al., [RewardBench: Evaluating Reward Models for Language Modeling](https://arxiv.org/abs/2403.13787), 2024.

[^zero]: Rajbhandari et al., [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054), 2019.

[^deepspeedzero]: Microsoft DeepSpeed, [ZeRO Tutorial](https://www.deepspeed.ai/tutorials/zero/).

[^fsdp]: PyTorch Docs, [FullyShardedDataParallel](https://docs.pytorch.org/docs/stable/fsdp.html).

[^lora]: Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), 2021.
