# B.2 Agentic RL ：、

> B.1  RL ：rollout、buffer、trainer、。： action  token，、、、，。

##  B.1 

B.2  vLLM/SGLang ， FSDP、ZeRO、TP、PP、EP 。 B.1 。B.2  Agentic RL  LLM RL 。

|            | B.1                                             | B.2 Agentic RL                                              |
| -------------- | ----------------------------------------------------------- | ------------------------------------------------------------------- |
| rollout  | prompt ， completion                  | ，、                  |
|        | token、mask、logprob、reward、policy version、rollout batch | episode、tool call、tool result、、 reward、loss mask |
|        | rollout batch ，                | 、 IO、、 GPU           |
|        | buffer 、、staleness、、        | 、、、、              |
|        | OpenRLHF、veRL、slime                                       | Relax、AReaL、Agent-R1、NeMo Gym                                    |

：B.1 “”；B.2 “Agent ，、、”。

## 

 9  GRPO ：，，。，， GPU 。

， bug  Agent 。，、、、。，。，—— IO，，。 GPU ，。

。 Agent ，？****。， rollout batch ？****。 GPU ，？**GPU **。“ GPU”， Agent 。。

## 

Agent ，。，。， `os.system("rm -rf /")` ， API key。——。：，、。

，Agent **（sandbox）。、。

### 

，：

|                    |                |  |                      |
| ---------------------- | ---------------------- | -------- | ---------------------------- |
| subprocess +   |                  | ~10 ms   | ，           |
| Docker             |  +  +  | ~100 ms  | ，       |
| MicroVM（Firecracker） |                  | ~125 ms  | ， |
| WebAssembly（Wasm）    |                | ~1 ms    | ，     |

**subprocess + **。 `rlimit`  CPU 、 `chroot` 、 `unshare` 。（），，：

```python
import subprocess, resource

def run_in_subprocess(code, timeout=10, max_memory=256 * 1024 * 1024):
    """：subprocess + """
    def set_limits():
        resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))
        resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))

    result = subprocess.run(
        ["python", "-c", code],
        timeout=timeout,
        preexec_fn=set_limits,
        capture_output=True, text=True,
    )
    return result
```

**Docker **。 Linux cgroups  namespace 、，（ `python:3.11-slim`、`node:20` ）。（ 100 ），：

```python
container = client.containers.run(
    "python:3.11-slim",
    command=f"python -c '{code}'",
    detach=True,
    mem_limit="512m",       # 
    cpu_quota=50000,         # CPU 
    network_mode="none",     # 
    remove=True,
)
```

**MicroVM（ Firecracker）** —— VM  Linux ， VM  VM。AWS Lambda  Fly.io 。 125 ， Docker ，。。

**WebAssembly（Wasm）**  WASI（WebAssembly System Interface）。 Wasm ，，。 1 ，， Python 。

### 

，。 Agent ——，（ Agent ，，）。（ Web Agent），，。

### 

 Docker ，。 1000  episode，，。**（warm container pool）—— N ，， 5 。

"Agent "。，Agent ，。

## 

### LLM RL vs. Agentic RL 

B.1  LLM RL 。 token ids、attention mask、response mask、old logprob、policy version、reward 。，：`prompt -> completion -> reward`。

Agentic RL 。 episode ，、、、。" Python bug"：，，，，——。

### 

 B.1  rollout batch ，Agentic RL ：

- ****：""
- ****：
- ****：，

（），JSON  SQLite 。（） Redis 、S3 。（MongoDB  DynamoDB）。 Agent，——（URL），， KB 。

，： GPU 。

## GPU 

### 

B.1  LLM RL  GPU ：， GPU 。Agentic RL ，。

 Agentic ：GPU  3 ， CPU  500 。 500  GPU 。：GPU 3 ，CPU 300 。，GPU  1%。 LLM RL ，LLM RL  rollout  training ， Agentic RL 。

### ：

 B.1 ：。 A ，GPU  B ； B ，GPU  C 。，GPU 。 GPU  1%  70-80%， 50-100 。

### 

""。 B.1  Rollout  Training 。：（GPU ）， Rollout  Training （Rollout ，Training ）。 Agentic RL ， B.1 。

，Agentic RL ——、、GPU ——。？ Relax 。

## ：Relax

Relax  AI Infra  Agentic RL ，（、、）Agentic RL 。、、。

### 

Relax  Ray Serve ：Actor、Rollout、Critic、Reference、Advantages、GenRM 。 Agentic RL —— GPU、 CPU、 CPU 。、，。

```
┌───────────────────────────────────────────────────────────────┐
│  Entrypoints:  train.py                                        │
├───────────────────────────────────────────────────────────────┤
│  Orchestration:  Controller () │ Service │ Registry    │
├───────────────────────────────────────────────────────────────┤
│  Components:  Actor │ Rollout │ Critic │ ActorFwd │ GenRM     │
├───────────────────────────────────────────────────────────────┤
│  Engine:  SGLang  │  │  │              │
├───────────────────────────────────────────────────────────────┤
│  Backends:  Megatron-LM () │ SGLang ()                 │
├───────────────────────────────────────────────────────────────┤
│  Distributed:  Ray Actor Groups │ DCS ()               │
└───────────────────────────────────────────────────────────────┘
```

 Megatron-LM， B.1  TP/PP/CP/EP 。 SGLang， Megatron Bridge 。

### TransferQueue：

 B.1 ：Rollout  Buffer，Training  Buffer 。 Buffer ——Rollout  batch ，Training 。：Rollout  Buffer ，Training  Buffer  GPU 。 Agentic RL ，，。

TransferQueue ：Rollout ，Training ， batch 。 Agentic RL ， completion， episode。 DCS（Distributed Checkpoint Service）——Training ，DCS  NCCL  Rollout ，，。

 Batch  Sample ，。

### 

Relax 。

*Collocate *，Actor  Rollout  GPU，。Rollout  batch， GPU  Training。 GPU ， on-policy——，Training 。

*Fully Async *， GPU ， TransferQueue ， DCS 。 `--max-staleness` ""—— 0  on-policy，。 B.1 ""； Agentic RL ""、，。

### 

**Loss mask。**  Agentic RL ： token  loss 。，，。"、"，""。Relax  _loss mask_ ： token  mask=1 ， token  mask=0 。

**。** `BaseInteractionEnv`  `reset` / `step` / `format_observation` ， Rollout 。。，，。

**。** ，，。Relax  Rollout  `image_data`， Training  `multimodal_train_inputs`，。

**。** RL  60-70%  Rollout 。 Rollout ，Relax ：

```bash
# 
curl -X POST http://controller:8000/scale \
  -d '{"target_engine_count": 4, "mode": "ray_native"}'

# （）
curl -X POST http://controller:8000/scale \
  -d '{"engine_urls": ["gpu-cluster-2:8000"], "mode": "external"}'
```

`external` —— GPU  Rollout，。

### 、

**。** Relax ：GRPO（ [8.1-8.2 ](/chapter09_grpo_rlvr/grpo-practice-and-mechanism)）、GSPO、SAPO  OPD（ [8.5 ](/chapter09_grpo_rlvr/on-policy-distillation)）。 Service  `ALGOS` 。

**。** Qwen3 （4B、30B-A3B MoE）、Qwen3-VL（）、Qwen3-Omni（） Qwen3.5。

**。** HealthManager （，）；Metrics Service  TensorBoard / WandB / ClearML；Apprise  Slack、、。 RL ，——， GPU 、、OOM 。，。

### 

|      |              |                               |  |        |
| -------- | ------------------ | --------------------------------- | ------ | ---------- |
| AReaL    |  &         | ，2.77x                 |      |      |
| Seer     | Moonshot AI (Kimi) | ，rollout  +74–97%    |      |        |
| Agent-R1 |              | MDP ，/       |      |    |
| NeMo Gym | NVIDIA             |  Agent                    |      |    |
| slime    |  /         | Megatron + SGLang，MoE    |      |    |
| Relax    |              | TransferQueue +  +  |      |  |

Relax  Agentic RL 。Seer ——，（divided rollout、context-aware scheduling、adaptive grouped speculative decoding） rollout ， GRPO  74–97%， on-policy （[arXiv:2511.14617](https://arxiv.org/abs/2511.14617)）。slime  SGLang 、Megatron ， GLM-4.5、Qwen3-30B-A3B、DeepSeek-R1  MoE  fp8 rollout、DeepEP ， MoE （[THUDM/slime](https://github.com/THUDM/slime)）。Relax  [arxiv.org/abs/2604.11554](https://arxiv.org/abs/2604.11554)， [github.com/redai-infra/Relax](https://github.com/redai-infra/Relax)。

## 

。，TRL  subprocess —— reward 。（SFT → DPO/GRPO → ），[ms-swift](https://github.com/modelscope/ms-swift)  ModelScope ，。（） veRL  OpenRLHF， Docker  asyncio 。 Agentic  Relax  AReaL 。 Agent ，Relax 。

：，，。

## ：nanoRLHF —  LLM RL 

：，，GPU 。 RL ，。[hyunwoongko/nanoRLHF](https://github.com/hyunwoongko/nanoRLHF)（181 stars）—— PyTorch + Triton  LLM RLHF ，、、 RL 。

nanoRLHF  nanoGPT：。 B.1 ：

```
nanorlhf/
├── nanotron/     # （3D parallelism、gradient accumulation、checkpoint）
├── nanovllm/     # （PagedAttention、KV cache、continuous batching）
├── nanoverl/     # RL （PPO trainer、reward、dataset、configs）
├── nanoray/      # （、）
├── nanosets/     # 
├── kernels/      # Triton kernel（fusion、）
└── eval/         # 
```

### ：nanotron

nanotron  B.1 "/"—— GPU 。 3D parallelism（ +  + ）、gradient accumulation、mixed precision training  checkpoint 。

：`nanotron/` 。：

- （`nanotron/parallel`）
- （`nanotron/pipeline`）
- 

### ：nanovllm

nanovllm  B.1 "/rollout "—— token。 PagedAttention（vLLM ）、KV cache  continuous batching。

：`nanovllm/` 。：

- PagedAttention  KV cache 
- continuous batching  GPU
- 

### RL ：nanoverl

nanoverl ， B.1  OpenRLHF/veRL 。 PPO ：rollout（ nanovllm ）→ reward  → advantage  → PPO clipped loss → （ nanotron ）。

：`nanoverl/trainer/` 。：

- PPO  fit()  actor、reference、rollout 
- KL （reference model ）
- reward （）

### 

，：

1. **`nanotron/`** — ，
2. **`nanovllm/`** — ， rollout 
3. **`nanoverl/`** —  RL  PPO ，"-"
4. **`nanoray/`** — ，

### 

```bash
# 
git clone https://github.com/hyunwoongko/nanoRLHF.git
cd nanoRLHF

# （ CUDA GPU）
pip install -e .
```

：

1. ** SFT **：`bash ./scripts/train_sft.sh`， loss、lr、throughput 
2. ** PPO trainer**： `nanoverl/trainer/`， rollout → reward → advantage → train 
3. ** B.1 **： nanoRLHF  OpenRLHF / veRL / slime ，
4. ** reward **： `nanoverl/reward/`  reward （、）， reward  RL 

nanoRLHF ， B.1 "rollout engine、training backend、weight sync、policy version"。 veRL  OpenRLHF ，。

## 

[^relax_paper]: Zhang L, Ning B, Yang R, et al. "[Relax: An Asynchronous Reinforcement Learning Engine for Omni-Modal Post-Training at Scale](https://arxiv.org/abs/2604.11554)." arXiv:2604.11554, 2026. [GitHub](https://github.com/redai-infra/Relax)

[^1]: HuggingFace Blog, "[Async RL Training Landscape — 16 Open-Source Libraries Compared](https://huggingface.co/blog/async-rl-training-landscape)", 2026.

[^2]: PyTorch Blog, "[A Primer on LLM Post-Training](https://pytorch.org/blog/a-primer-on-llm-post-training/)", 2025.

[^3]: AReaL Team. "[AReaL: Async RL for Language Reasoning](https://arxiv.org/abs/2505.24298)." arXiv:2505.24298, 2025. [GitHub](https://github.com/inclusionAI/AReaL)

[^4]: Hou L et al. "[Seer: Online Context Learning for Fast Synchronous LLM Reinforcement Learning](https://arxiv.org/abs/2511.14617)." arXiv:2511.14617, 2025.

[^5]: Ko H. "[nanoRLHF: From-scratch journey into how LLMs and RLHF really work](https://github.com/hyunwoongko/nanoRLHF)." GitHub, 2025.
