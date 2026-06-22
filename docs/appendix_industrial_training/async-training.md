---
search: false
---

# ：（ B.1）

> 。 [B.1 RL ：、](./rl-infrastructure) “”。，。

> LLM RL ：**（），， GPU 。** 。
>
> ：**？** ，，——、。

## ：，GPU 99% 

 TRL  GRPO ，：

```
①  512       ← ，， GPU 
②  loss、  ← ，， GPU 
③  512     ← ①
```

①②—— GPU 。？

 H100 （bf16，vLLM ）， GRPO （G=8 × 64 prompts = 512  rollout）：

|  |  token  | 7B  | 32B  |
| ------------ | ----------- | ----------- | ------------ |
| 2K token     | ~100      | 3       | 14       |
| 8K token     | ~400      | 11      | **56 **  |
| 32K token    | ~1600     | 45      | **3.7 ** |

（ + ） 30 。

，** GPU  99% **。""：GRPO  prompt  8 ，。

## 

### （Synchronous）

。 GPU，，，，。

```
:  [ b0] [ b0] [ b1] [ b1] [ b2] ...
              ↑      ↑      ↑ 
```

：，GPU 。：。

::: tip ：Seer
Seer（Moonshot AI / Kimi）——，""。： prompt ，。，Seer ：divided rollout（）、context-aware scheduling（）、adaptive grouped speculative decoding（）。 rollout  74–97%， 75–93%， on-policy —— GRPO 。Seer  Mooncake  KV cache + vLLM + Megatron， [arXiv:2511.14617](https://arxiv.org/abs/2511.14617)。
:::

### （Colocated）

 GPU，。（ FSDP  vLLM ），。，。

```
:  [ b0] [] [ b0] [] [ b1] [] ...
```

： GPU。：。。

### （Disaggregated / ）

。 GPU，（Buffer）：

```
 GPU :  [ b0] [ b1] [ b2] [ b3] ...
                ↓ buffer  ↓ buffer
 GPU :     [ b0]  [ b1]  [ b2] ...
                ↑     ↑ 
```

 GPU  buffer， GPU  buffer 。。 RL 。

|  | GPU    |        |                |
| ---- | ---------- | ---------------- | ---------------------- |
|  |        |              | 、           |
|  |        | （） | 、GPU  |
|  |  | ****           |          |

：********。

## ： GPU

 GPU ， GPU ？：，。

### ？

|                 |     |  |      |
| ----------------------- | --------- | ---------- | ------------ |
| NCCL          | 100-500ms |    |    |
| NCCL （veRL）   | ~20ms     |    |      |
| GPU （NeMo-RL） |       |    |    |
|  LoRA adapter       | **<1ms**  | ~50MB      | **** |
| （PRIME-RL）  |       |    |        |

****： LoRA ， adapter （~50MB）， 1 。 LoRA + 。

### ？

 GPU  32K token ，，。：

1. ****（PipelineRL）： token （~1-10ms）。。
2. ****（veRL、AReaL）：，。。
3. ****（SkyRL、SLIME）：，。 token。
4. ** batch **（NeMo-RL、Tunix）：，。

 LoRA  <1ms，。

## ：

， GPU 。

： GPU  100 ， 20ms， GPU  99 。""，""，—— off-policy （ [ 5 ](/chapter05_policy_gradient/intro) ）。

？。 1-2 ， 10 。：

### ：，

。（ 5 ）。，，。

### ：，

—— 2 。"" 2 。，。

### ：，

：，。""（Truncated IS）。，，。。

### 

。 PRIME-RL ： +  + 。

|        |                |      |
| ---------- | ---------------------- | ---------- |
|    | PipelineRL、TorchForge |    |
|  | AReaL                  |    |
|    | veRL、ROLL             |  |
|    | PRIME-RL               |        |

：** + **，。

## 

[^1]: HuggingFace Blog, [Async RL Training Landscape — 16 Open-Source Libraries Compared](https://huggingface.co/blog/async-rl-training-landscape), 2026. 。

[^2]: PyTorch Blog, [A Primer on LLM Post-Training](https://pytorch.org/blog/a-primer-on-llm-post-training/), 2025.  Meta 。

[^3]: OpenRLHF, [OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework](https://arxiv.org/abs/2405.11143), EMNLP 2025 Demo.

[^4]: veRL Project, [HybridFlow: A Flexible and Efficient RL Training Framework](https://github.com/verl-project/verl),  / .
