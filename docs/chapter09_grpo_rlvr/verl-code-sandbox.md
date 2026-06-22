# 9.7 ： veRL 

 OPD ， teacher 。 RLVR ，：****。

：，。，；，。 RLVR ， reward ——，。

 veRL  PPO 。8.7  GSM8K  veRL，， reward ：，。

 veRL Code Sandbox [^volcengine-verl-code-sandbox]，：

- ****：Eurus-2-RL-Data  + Qwen2.5  + PPO（GAE advantage ）。
- ****：filter  prompt、 1000 。
- **Reward **： → / → 。
- ****： EvalScope  GSM8K、HumanEval、LiveCodeBench ， RL 。

 VKE  + SandboxFusion 。** GPU **：，/，。 Agent  [10.5  rLLM  DeepCoder Agent](../chapter10_agentic_rl/rllm-deepcoder-lab)， AgentFlow  sandbox cookbook； verifier  veRL 。

```mermaid
flowchart LR
    P[" prompt"] --> M[" πθ"]
    M --> C[""]
    C --> S["Verifier\n + "]
    S --> R["reward\npass/fail "]
    R --> T["veRL Trainer\nPPO / GRPO "]
    T --> M

    style S fill:#e8f5e9,stroke:#2e7d32
    style R fill:#fff3e0,stroke:#f57c00
```

##  RLVR

""。，，，Reward Model 。

。 `two_sum(nums, target)`：

```python
def two_sum(nums, target):
    ...
```

：

```python
assert two_sum([2, 7, 11, 15], 9) == [0, 1]
assert two_sum([3, 2, 4], 6) == [1, 2]
assert two_sum([3, 3], 6) == [0, 1]
```

，，reward 。，，reward 。""。

 RLVR  reward ：

|           |                          |  reward |
| ------------- | -------------------------------- | ----------- |
|       | 、 | 0.0–0.2     |
| / |  import                | 0.0–0.3     |
|       |                  | 0.0–1.0     |

。。

## 

### 

** GPU**（24GB ， RTX 3090 / 4090 / A5000）****：

|                      |  |          |           |
| ------------------------ | ------ | ---------------- | ----------------- |
| Qwen2.5-Coder-0.5B      | 0.5B   |  + vLLM      | ~18 GB（）    |
| Qwen2.5-Coder-1.5B      | 1.5B   | LoRA + vLLM      | ~20 GB（）    |
| Qwen2.5-Coder-7B        | 7B     |          | ~80 GB（A100 ） |

 8.7 ，PPO  Actor、Critic（） Reference（）， vLLM ， SFT 。0.5B  + 。

###  veRL

 8.7  veRL，。：

```bash
# 
conda create -n verl python==3.10 -y
conda activate verl

#  PyTorch（CUDA 12.x）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

#  veRL
git clone https://github.com/volcengine/verl.git
cd verl
pip install -e .

#  vLLM（）
pip install vllm==0.8.3

#  Flash Attention
pip install flash-attn --no-build-isolation
```

### 

 [Eurus-2-RL-Data](https://huggingface.co/datasets/PRIME-RL/Eurus-2-RL-Data) ， PRIME-RL ，。：、。 train  validation  split。

， veRL  parquet ：

```python
# download_data.py
from datasets import load_dataset
from pathlib import Path

output_dir = Path.home() / "data" / "eurus2"
output_dir.mkdir(parents=True, exist_ok=True)

ds = load_dataset("PRIME-RL/Eurus-2-RL-Data")

for split in ["train", "validation"]:
    df = ds[split].to_pandas()
    df.to_parquet(output_dir / f"{split}.parquet")
    print(f"{split}: {len(df)} ，: {list(df.columns)}")
```

，`~/data/eurus2/` 。：

```python
import pandas as pd

df = pd.read_parquet("~/data/eurus2/train.parquet")
print(f": {len(df)}")
print(f": {list(df.columns)}")
print(f"prompt : {df['prompt'].str.len().mean():.0f}")
print(f"prompt : {df['prompt'].str.len().max():.0f}")

# 
example = df.iloc[0]
print(f"\n===  0 ===")
print(f"prompt:\n{example['prompt'][:500]}...")
print(f"\ntests:\n{str(example.get('tests', ''))[:300]}...")
```

，。Eurus-2-RL-Data  prompt （ `max_prompt_length=512`）， filter，。， 1000 ：

```python
# prepare_data.py —  + 
import pandas as pd
from pathlib import Path

data_dir = Path.home() / "data" / "eurus2"

df = pd.read_parquet(data_dir / "train.parquet")
print(f": {len(df)} ")

#  prompt  512 token 
# ：1 token ≈ 4 
MAX_PROMPT_CHARS = 512 * 4
df = df[df["prompt"].str.len() < MAX_PROMPT_CHARS]
print(f" prompt : {len(df)} ")

#  1000 （ random_state ）
n_samples = min(1000, len(df))
subset_df = df.sample(n=n_samples, random_state=42)
subset_df.to_parquet(data_dir / "train1000.parquet")
print(f" {len(subset_df)}  train1000.parquet")
```

，`train1000.parquet` 。：

|            |                                  |                                         |
| -------------- | ------------------------------------ | ------------------------------------------- |
| `prompt`       |                | "Write a function `two_sum(nums, target)`…" |
| `entry_point`  |                      | `"two_sum"`                                 |
| `tests`        | （Python assert ） | `assert two_sum([2,7,11,15], 9) == [0,1]`   |
| `ground_truth` | ，           | —                                           |

 `prompt`，reward  `tests` 。 RLVR ——**reward ，**。

## Reward 

8.7  GSM8K reward ，。： markdown ，，、。

 8.7 。 reward 。

### ：

 markdown 。 Python ：

```python
import re


def extract_code(response: str) -> str:
    """ Python 。

    ：
        "：\n```python\ndef two_sum(nums, target):\n    ..."
     ```python  ``` 。
    ，（）。
    """
    match = re.search(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()
```

，`extract_code` ——，reward  0。，。

### ：

，。：、、。 `exec`， `while True: pass` 。

 `multiprocessing.Process` ，：

```python
import multiprocessing as mp


def run_tests(code: str, tests: str, queue: mp.Queue) -> None:
    """。

    ：
    1. exec(code, namespace) — ，
    2. exec(tests, namespace) — ，
    3. ， queue  passed=True
    4. ，

    ： exec  subprocess， Python
     namespace，。
     Docker ，。
    """
    namespace = {}
    try:
        exec(code, namespace)
        exec(tests, namespace)
        queue.put({"passed": True, "error": ""})
    except Exception as exc:
        queue.put({"passed": False, "error": repr(exc)})
```

### ：

，：

```python
def score_code(response: str, tests: str, timeout_s: float = 3.0) -> float:
    """： →  → 。

     0/1  reward：=1.0，=0.0。
     3 （）， kill  0。
    """
    code = extract_code(response)

    #  reward：
    if not code:
        return 0.0

    # 
    queue = mp.Queue()
    process = mp.Process(target=run_tests, args=(code, tests, queue))
    process.start()
    process.join(timeout=timeout_s)

    # ：kill 
    if process.is_alive():
        process.kill()
        process.join()
        return 0.0

    if queue.empty():
        return 0.0

    result = queue.get()
    return 1.0 if result["passed"] else 0.0
```

 3 。 LeetCode  1 ，3 。，， 0 。

### ： veRL  reward 

veRL  reward 。`compute_score`  veRL ：

```python
from typing import Any


def compute_score(reward_input: dict[str, Any], **kwargs) -> dict[str, float]:
    """veRL reward 。

    veRL  reward_input ，：
    - response: 
    - extra_info: （ tests ）
    - ground_truth: （）

    ：
    - overall: PPO 
    - pass_rate: （）
    - format: （）
    """
    response = reward_input["response"]
    extra_info = reward_input.get("extra_info", {})
    tests = extra_info.get("tests", "")

    score = score_code(response, tests)
    return {
        "overall": score,
        "pass_rate": score,
        "format": 1.0 if extract_code(response) else 0.0,
    }
```

### 

 reward ：

```python
# code_reward.py
#  RLVR  reward 
# ：veRL  custom_reward_function 

import multiprocessing as mp
import re
from typing import Any

REWARD_NAME = "code_rlvr"
REWARD_TYPE = "sequential"


def extract_code(response: str) -> str:
    """ Python 。"""
    match = re.search(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


def run_tests(code: str, tests: str, queue: mp.Queue) -> None:
    """。"""
    namespace = {}
    try:
        exec(code, namespace)
        exec(tests, namespace)
        queue.put({"passed": True, "error": ""})
    except Exception as exc:
        queue.put({"passed": False, "error": repr(exc)})


def score_code(response: str, tests: str, timeout_s: float = 3.0) -> float:
    """： →  → 。"""
    code = extract_code(response)

    if not code:
        return 0.0

    queue = mp.Queue()
    process = mp.Process(target=run_tests, args=(code, tests, queue))
    process.start()
    process.join(timeout=timeout_s)

    if process.is_alive():
        process.kill()
        process.join()
        return 0.0

    if queue.empty():
        return 0.0

    result = queue.get()
    return 1.0 if result["passed"] else 0.0


def compute_score(reward_input: dict[str, Any], **kwargs) -> dict[str, float]:
    """veRL reward 。"""
    response = reward_input["response"]
    extra_info = reward_input.get("extra_info", {})
    tests = extra_info.get("tests", "")

    score = score_code(response, tests)
    return {
        "overall": score,
        "pass_rate": score,
        "format": 1.0 if extract_code(response) else 0.0,
    }
```

 reward ：**，**。，，reward  0。 RM 。

## Prompt 

，prompt 。， verifier 。

````text
You are a competitive programming assistant.

Solve the following problem in Python.
Return only one Python code block.

Function name: {entry_point}

Problem:
{problem_statement}

Your answer:
```python
```
````

 Chat ，； base coder， prompt。。

## 

 8.7  veRL PPO ，。，： Eurus-2-RL-Data、reward 、`max_response_length`  256  512（）。

 8.7 ：，，。

```bash
#!/bin/bash
# run_qwen_coder_ppo_single_gpu.sh
# PPO | Eurus-2  |  | Qwen2.5-Coder-0.5B

set -xeuo pipefail

# ====================  ====================
# ：Qwen2.5-Coder ， Instruct 
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-Coder-0.5B-Instruct}
CRITIC_MODEL_PATH=${CRITIC_MODEL_PATH:-$MODEL_PATH}  # Critic 

# 
NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-1}

# 
# batch_size=128  128  prompt 
# mini_batch=64  PPO  2  mini-batch（128/64）
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}

# 
#  max_response_length  GSM8K （512 vs 256）
# 
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}

# 
# Actor lr  Critic lr ， PPO 
# Actor ，Critic  value function
ACTOR_LR=${ACTOR_LR:-1e-6}
CRITIC_LR=${CRITIC_LR:-1e-5}

# 
# vLLM ，=1
ROLLOUT_TP=${ROLLOUT_TP:-1}
# vLLM ，
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.4}
#  prompt （PPO ）
ROLLOUT_N=${ROLLOUT_N:-1}

# 
TOTAL_EPOCHS=${TOTAL_EPOCHS:-20}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-5}

# 
TRAIN_FILE=${TRAIN_FILE:-$HOME/data/eurus2/train1000.parquet}
VAL_FILE=${VAL_FILE:-$HOME/data/eurus2/validation.parquet}

EXPERIMENT_NAME=${EXPERIMENT_NAME:-coder_ppo_eurus2_$(date +%Y%m%d_%H%M)}
# ====================  ====================

# ----  ----
# filter_overlong_prompts=True:  max_prompt_length 
# truncation='error': ，
DATA=(
    algorithm.adv_estimator=gae
    data.train_files="['$TRAIN_FILE']"
    data.val_files="['$VAL_FILE']"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

# ----  ----
# enable_gradient_checkpointing=True: ，
# use_remove_padding=True:  padding token 
MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

# ---- Actor  ----
# clip_ratio=0.2: PPO ，
# param_offload=False: （ CPU ）
ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384
    actor_rollout_ref.actor.clip_ratio=0.2
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

# ---- Rollout  ----
# name=vllm:  vLLM  continuous batching 
# gpu_memory_utilization=0.4: vLLM  40% ，
ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
)

# ---- Reference  ----
# param_offload=True: Reference ， CPU 
REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

# ---- Critic  ----
# Critic  Actor （1e-5 vs 1e-6）
# Critic  Actor  advantage 
CRITIC=(
    critic.model.path="$CRITIC_MODEL_PATH"
    critic.model.use_remove_padding=True
    critic.model.enable_gradient_checkpointing=True
    critic.optim.lr=${CRITIC_LR}
    critic.use_dynamic_bsz=True
    critic.ppo_max_token_len_per_gpu=16384
    critic.fsdp.param_offload=False
    critic.fsdp.optimizer_offload=False
)

# ---- Trainer  ----
TRAINER=(
    trainer.balance_batch=True
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=verl_ppo_code
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
)

# ----  ----
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${CRITIC[@]}" \
    "${TRAINER[@]}" \
    "$@"
```

### 

 8.7  GSM8K  PPO ，：

|                     | GSM8K（8.7 ） | （） |                                    |
| ------------------------- | --------------- | ---------------- | -------------------------------------- |
|                     | GSM8K     | Eurus-2-RL-Data  |          |
| reward                | `gsm8k_reward`  | `code_reward`    |  +  +              |
| `max_response_length`     | 256             | 512              |              |
|                   | Qwen2.5-0.5B    | Qwen2.5-Coder    |  coder           |

（、clip_ratio、GAE ） 8.7 —— PPO ，。

###  8.7 

 8.7 ，PPO ：

| 8.7  |                           |                                |
| ---------- | --------------------------------- | ---------------------------------- |
| Actor      | `actor_rollout_ref.actor.*`       | ，     |
| Reference  | `actor_rollout_ref.ref.*`         |  SFT ， KL       |
| Critic     | `critic.*`                        | ，GAE  advantage |
| RM/Reward  | `code_reward.py:compute_score`    | ： +  +        |

：8.7 （），（ →  → ）。reward  0/1 ， reward 。

## 

### ：

```bash
chmod +x run_qwen_coder_grpo_single_gpu.sh
bash run_qwen_coder_grpo_single_gpu.sh
```

### ：

```bash
#  1.5B coder 
MODEL_PATH=Qwen/Qwen2.5-Coder-1.5B-Instruct \
TRAIN_BATCH_SIZE=64 \
PPO_MINI_BATCH_SIZE=16 \
bash run_qwen_coder_grpo_single_gpu.sh
```

```bash
# （8 ）
NNODES=1 NDEVICES_PER_NODE=8 \
TRAIN_BATCH_SIZE=1024 \
PPO_MINI_BATCH_SIZE=256 \
ROLLOUT_TP=2 \
bash run_qwen_coder_grpo_single_gpu.sh
```

Ray  `main_ppo` 。， worker  GPU ； Ray ，。

### 

，：

```
[Step 1]  train | reward/overall=0.03 | reward/pass_rate=0.03 | reward/format=0.15 | kl=0.000
[Step 5]  val   | reward/overall=0.08 | reward/pass_rate=0.08
[Step 6]  train | reward/overall=0.12 | reward/pass_rate=0.12 | reward/format=0.45 | kl=0.002
[Step 10] val   | reward/overall=0.21 | reward/pass_rate=0.21
```

 `format`  `pass_rate` ——""，""。 RLVR 。

## 

### 

|               |                |               |
| ----------------- | ---------------------- | --------------------- |
| `reward/pass_rate`|                |  0    |
| `reward/format`   |  pass_rate     | （） |
| `kl`              |                |               |
| `actor_loss`      |  0.5~1.0     |  >10  NaN     |
| `response_length` |          |  reward     |

###  RLVR 

** 1：（step 1~10）**。`pass_rate`  0， `format` 。" \`\`\`python "，。`kl`  0。

** 2：（step 10~40）**。`pass_rate` 。，，。 PPO 。

** 3：（step 40+）**。`pass_rate` 。——，。

### 

（Qwen2.5-7B-Instruct-1M，Eurus-2-RL-Data  1000 ，130 steps PPO）[^volcengine-verl-code-sandbox]， [EvalScope](https://github.com/modelscope/evalscope)  benchmark ：

|                                       | GSM8K | HumanEval | LiveCodeBench |
| ----------------------------------------- | ----- | --------- | ------------- |
| Qwen2.5-7B-Instruct-1M（）            | 0.82  | 0.59      | 0.50          |
| Qwen2.5-7B-Instruct-1M-step130（RL ）   | 0.83  | 0.59      | 0.53          |

：

- **LiveCodeBench **（0.50 → 0.53），——RL 。
- **GSM8K **（0.82 → 0.83）， RL 。
- **HumanEval **（0.59）， benchmark ，1000 。

 RL ，，，。，。

> ****： GPU 。、，，。

## 

， checkpoint ， PPO 。

### Checkpoint 

veRL  FSDP ， checkpoint  GPU 。 HuggingFace ：

```bash
python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir /path/to/checkpoints/global_step_20/actor \
    --target_dir ./merged_model
```

### EvalScope 

 [EvalScope](https://github.com/modelscope/evalscope) ：

```bash
#  EvalScope
pip install evalscope

# （HumanEval + LiveCodeBench）
evalscope eval \
    --model ./merged_model \
    --datasets humaneval livecodebench \
    --limit 100

# （）
evalscope eval \
    --model ./merged_model \
    --datasets gsm8k \
    --limit 100
```

：

- ** test **：，。
- ** baseline**： RL ， PPO 。
- ** benchmark **： HumanEval ，LiveCodeBench 。

## 

，：

|                     |  | 8  |                                  |
| ----------------------- | ---- | ---- | ------------------------------------ |
| `NDEVICES_PER_NODE`     | 1    | 8    | GPU                              |
| `TRAIN_BATCH_SIZE`      | 128  | 1024 |  batch（FSDP ）      |
| `PPO_MINI_BATCH_SIZE`   | 64   | 256  |                                  |
| `ROLLOUT_TP`            | 1    | 2    | vLLM                       |
| `ROLLOUT_GPU_MEM_UTIL`  | 0.4  | 0.6  |                |

、clip_ratio、GAE ****——，。

##  10.5 DeepCoder 

 [10.5](../chapter10_agentic_rl/rllm-deepcoder-lab) ： sandbox reward 。：

|      |  |                                        |
| -------- | ---- | ------------------------------------------ |
| 9.7  | veRL |  verifier  PPO/GRPO      |
| 10.5     | rLLM |  DeepCoder cookbook  Agentic   |

， 10.5。 veRL， RLVR ， data、reward、trainer 。

## 

，：

- 。
- reward ， rollout。
- reward ：、、。
-  reward， eval set  Pass@1。
- ，。

 RL 、；。 verifier ， PPO/GRPO 。

[^volcengine-verl-code-sandbox]: ，"veRL Code Sandbox "，https://www.volcengine.com/docs/6460/1756203
