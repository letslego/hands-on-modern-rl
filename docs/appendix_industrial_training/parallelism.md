---
search: false
---

# ：（ B.1）

> 。 [B.1 RL ：、](./rl-infrastructure) “”。，。

> PPO  4 （Actor、Critic、Reference、Reward Model）。7B  FP16  ~14GB，4  ~56GB， A100（80GB）。。
>
> ：**，？** ，， RL 。

## 

### （DP）

。 GPU ， batch， AllReduce 。

：。7B  FP16  ~14GB  +  + ， ~56GB，。

### （TP）

 GPU 。 4096×4096 ，4  GPU  1024×4096，。

-  NVLink （）
- Megatron-LM 
- （ 8×H100）

### （PP）

。GPU 0  1-10 ，GPU 1  11-20 ，。

""—— GPU 。（Micro-batching），。

### （EP）

 MoE（）。 GPU ，Router  token 。Mixtral、DeepSeek-V3 。

****：70B  DP+TP+PP 。MoE  EP。

|  |    |                 |             |
| ---- | -------- | --------------------- | ------------------- |
| DP   |      |  AllReduce        |     |
| TP   |  |  forward/backward | （ NVLink） |
| PP   |      |             |               |
| EP   |  | Router            | MoE             |

## 

""。""：

**FSDP（Fully Sharded Data Parallel）**：PyTorch 。、、 GPU，。，。

**DeepSpeed ZeRO**： FSDP，。ZeRO-1 ，ZeRO-2 ，ZeRO-3 。ZeRO-3 ，。

 FSDP  ZeRO  TP/PP 。

## 

|    |     |  |      |                              |
| ------ | ------- | ---- | ---------- | ---------------------------------- |
| BF16   |     |    |          | ****（A100/H100 ） |
| FP16   |     |    |  |  loss scaling                    |
| FP32   |     |    |        |                        |
| FP8    | 1/4     |  |        |                          |
| INT8/4 | 1/4~1/8 |  |        | ****                       |

： BF16，（INT4/INT8）。

## RL 

RL （PPO/GRPO），：

**Rollout **：GRPO  k=16  prompt  16 ， GPU 。vLLM  PagedAttention  2-4x。

**Training **： GPU。

** GPU **：Rollout ，。——Rollout  GPU ， [B.1 RL ](./rl-infrastructure) 。

****：

|                    |                                 |            |
| ---------------------- | ----------------------------------- | -------------- |
| Reference      | Ref ， Actor  | ~25%           |
| LoRA Rollout           |  LoRA adapter           | ~50%           |
| Gradient Checkpointing |               | ~40%（） |

## ：MoE  PRM 

**MoE **：MoE ，（Megatron）（vLLM） Router ， token  Expert A、 Expert B——。DeepSeek-V3.2  Keep Routing：，。

**PRM **：Process Reward Model ，。 GPU  PRM scoring。（ PRIME-RL）。

## 

[^1]: HuggingFace Blog, [Async RL Training Landscape — 16 Open-Source Libraries Compared](https://huggingface.co/blog/async-rl-training-landscape), 2026.

[^2]: PyTorch Blog, [A Primer on LLM Post-Training](https://pytorch.org/blog/a-primer-on-llm-post-training/), 2025.

[^3]: OpenRLHF, [OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework](https://arxiv.org/abs/2405.11143), EMNLP 2025 Demo.

[^4]: DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437), 2024.
