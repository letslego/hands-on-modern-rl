# 9.5 RLVR：

 9.3  9.4 ， GRPO  Critic、DAPO  RL  SFT。：****。，（）。RLVR（Reinforcement Learning with Verifiable Rewards）：**， RM，**。

 RLVR ， RM，， RLVR ，。

## RLVR 

 RLHF  Reward Model 。RM ，、 reward hacking、。RLVR ：**， RM？**

```mermaid
flowchart TD
    subgraph rlhf ["RLHF "]
        R1["Prompt"] --> R2[""]
        R2 --> R3["Reward Model \n（ RM）"]
        R3 --> R4[""]
    end

    subgraph rlvr ["RLVR "]
        V1["Prompt"] --> V2[""]
        V2 --> V3["\n（、、）"]
        V3 --> V4[""]
    end

    R3 -.->|"：、、 hack"| COST1["💸 \n📊 RM \n🤖 Reward Hacking"]
    V3 -.->|"：、、 hack"| COST2["✅ \n✅ \n✅ "]

    style R3 fill:#fce4ec,stroke:#c62828
    style V3 fill:#e8f5e9,stroke:#2e7d32
```

###  RLHF 

|              | RLHF                       | RLVR                         |
| ---------------- | -------------------------- | ---------------------------- |
| ****     | （） | （）             |
| ****     | （）       | （）             |
| ****     |              |                      |
| ****     | （、）     | （、、） |
| ****   |  RM              | （）     |
| ** Hack ** | （ RM ） | （）           |

## ：RLVR 

：

|        |    |                         |
| ---------- | ---------- | --------------------------- |
|        |    | `\boxed{42}` ==     |
|        |    |  + test case  |
|    |  | Lean/Coq          |
|  |    | BLEU/COMET              |

 RLVR 。：****（）、****（）、****（，）。

""。： $\frac{22}{7}$， $3.1428...$，？ $(x+1)(x-2)$， $x^2 - x - 2$，？。，（）（/），。

## 1-Shot RLVR： RL

，ICLR 2025  RLVR ** 1 **。，，RL 。

 RLVR ——，RLVR ""，。，——RL ""，。

。 RLVR ""——。，（、、）。 DeepSeek-R1-Zero —— SFT ， RL ，""。

<details>
<summary>： RLVR  1 ，？</summary>

1 ""，，、。：

- ****： 1 ，""。，。
- ****：，，。
- ****：1 。。

1-Shot RLVR —— RL ""，""。 RL  LLM 。

</details>

## ： RLVR 

 RLVR 。。

， MATH  0.6B  Qwen3 ：，，， GRPO 。 200 ， GPU 。

 Sebastian Raschka  [reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch)  Chapter 6  RLVR GRPO —— RLVR 。，** RLVR " + GRPO"**。

### RLVR 

RLVR  GRPO ， RM：

```mermaid
flowchart TD
    subgraph Rollout ["Rollout："]
        P[" Prompt"]
        M[" π_θ"]
        P --> M
        M -->|" 1"| R1[" 1"]
        M -->|" 2"| R2[" 2"]
        M -->|"..."| R3[" G"]
    end
    subgraph Reward ["Reward："]
        R1 --> V["\nextract + grade"]
        R2 --> V
        R3 --> V
        V -->|"=1, =0"| Scores["reward_1, reward_2, ..., reward_G"]
    end
    subgraph Train ["Train：GRPO "]
        Scores --> Adv["advantage = (r - mean) / std"]
        Adv --> Loss["pg_loss = -(advantage × log_prob).mean()"]
        Loss --> Update[" θ → θ'"]
    end
    Update -->|""| M
```

：

- **Rollout **：， $π_θ$  $G$ （ $G=4$）。 `\boxed{}` 。
- **Reward **： `\boxed{}` ，。 1 ，（） 0 。
- **Train **： GRPO  advantage，。

### ：

RLVR ""。：，。

```python
import re

def extract_boxed_answer(text: str) -> str | None:
    """ \\boxed{...} 。

     \\boxed{} 。
    ， None（reward = 0）。
    """
    match = re.search(r"\\boxed\{([^}]*)\}", text)
    if match:
        return match.group(1).strip()
    return None

def grade_answer(predicted: str, ground_truth: str) -> bool:
    """。

    ： + 。
    （、）。
    """
    predicted = predicted.strip().replace(" ", "")
    ground_truth = ground_truth.strip().replace(" ", "")
    if predicted == ground_truth:
        return True
    # （ "22/7" vs "3.1428..." ）
    try:
        return abs(float(predicted) - float(ground_truth)) < 1e-6
    except ValueError:
        return False

def reward_rlvr(response: str, ground_truth: str) -> float:
    """RLVR ： + 。

     RLVR —— RM，，
     0/1 。
    """
    predicted = extract_boxed_answer(response)
    if predicted is None:
        return 0.0  # ， 0 
    return float(grade_answer(predicted, ground_truth))
```

：

- `extract_boxed_answer()`  `\boxed{}` 。，reward  0——，。
- `grade_answer()` ，。（ [reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch)  `sympy` ），。
- 、、—— RLVR  RM 。

### GRPO 

， " →  → GRPO " 。

```python
import torch
import torch.nn.functional as F


def compute_grpo_loss(model, tokenizer, prompt, ground_truth,
                      device, num_rollouts=4, max_new_tokens=512,
                      temperature=0.8):
    """ GRPO ：rollout → reward → compute loss。

    ：
        model: 
        tokenizer: 
        prompt: 
        ground_truth: 
        num_rollouts: （GRPO ）
        max_new_tokens: 
        temperature: 

    ：
        dict:  loss、rewards、advantages 
    """
    roll_rewards, rollout_data = [], []

    # ====================  1: Rollout ====================
    #  num_rollouts 
    with torch.no_grad():
        for _ in range(num_rollouts):
            input_ids = torch.tensor(
                tokenizer.encode(prompt), device=device
            ).unsqueeze(0)
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
            )
            # （ prompt）
            response = tokenizer.decode(
                output_ids[0, input_ids.shape[1]:],
                skip_special_tokens=True,
            )
            #  reward：=1, =0
            reward = reward_rlvr(response, ground_truth)
            roll_rewards.append(reward)
            rollout_data.append((output_ids[0], input_ids.shape[1]))

    # ====================  2: GRPO Advantage ====================
    # ：
    # advantage = (reward - mean) / std
    rewards = torch.tensor(roll_rewards, device=device)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    #  advantage  0 （），
    if torch.allclose(advantages, torch.zeros_like(advantages), atol=1e-8):
        return {"loss": 0.0, "loss_tensor": None, "rewards": roll_rewards}

    # ====================  3:  log prob ====================
    roll_logps = []
    for token_ids, prompt_len in rollout_data:
        logits = model(token_ids.unsqueeze(0)).logits.squeeze(0).float()
        logprobs = torch.log_softmax(logits, dim=-1)
        #  response  log prob
        targets = token_ids[1:]
        selected = logprobs[:-1].gather(1, targets.unsqueeze(-1)).squeeze(-1)
        roll_logps.append(selected[prompt_len - 1:].sum())

    logps = torch.stack(roll_logps)

    # ====================  4:  loss ====================
    # pg_loss = -(advantage × log_prob).mean()
    # advantage > 0 ，advantage < 0 
    pg_loss = -(advantages.detach() * logps).mean()

    return {
        "loss": pg_loss.item(),
        "loss_tensor": pg_loss,
        "rewards": roll_rewards,
        "advantages": advantages.tolist(),
    }


def train_rlvr(model, tokenizer, train_data, device,
               steps=100, num_rollouts=4, lr=1e-5, **kwargs):
    """RLVR 。

    ：
        train_data: ， "problem"  "answer"
        steps: 
        num_rollouts: GRPO 
        lr: 
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for step in range(steps):
        example = train_data[step % len(train_data)]
        prompt = f"Solve the following problem. Put your final answer within "
                 f"\\boxed{{}}.\n\nProblem: {example['problem']}"

        stats = compute_grpo_loss(
            model, tokenizer, prompt, example["answer"],
            device, num_rollouts=num_rollouts, **kwargs,
        )

        if stats["loss_tensor"] is not None:
            optimizer.zero_grad()
            stats["loss_tensor"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        reward_avg = sum(stats["rewards"]) / len(stats["rewards"])
        if (step + 1) % 5 == 0:
            print(f"Step {step+1:3d} | loss={stats['loss']:.4f} | "
                  f"reward_avg={reward_avg:.3f}")

    return model
```

：

- `compute_grpo_loss()`  GRPO ：rollout → advantage → log prob → loss。 [reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch) —— GRPO 。
- **reward ， RM。** `reward_rlvr()`  + ，， reward hacking。
- **all-zero advantage 。**  rollout ，advantage  0， 0。，（、）。
-  GRPO（ KL ）， DAPO  Dr. GRPO ： KL 。

### 

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# （0.6B ）， GPU 
model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# MATH （）
train_data = [
    {"problem": "What is the value of $x$ if $2x + 3 = 11$?",
     "answer": "4"},
    {"problem": "Compute $\\sum_{k=1}^{10} k$.", "answer": "55"},
    # ... 
]

model = train_rlvr(
    model=model,
    tokenizer=tokenizer,
    train_data=train_data,
    device=model.device,
    steps=100,
    num_rollouts=4,
    lr=1e-5,
    max_new_tokens=512,
)
```

### 

 RLVR + GRPO 。 [reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch)  veRL/OpenRLHF ，：

|       |                 |  RLVR                                 |
| --------- | --------------------------- | ----------------------------------------------- |
|     |  +        | sympy 、LaTeX 、          |
|   | `model.generate()`  | continuous batching、KV cache、vLLM/SGLang      |
| GRPO  |  KL （）        | clip、KL 、length reward、Dr. GRPO    |
|     |                         | FSDP / Megatron、 GPU、gradient accumulation  |
|       |  reward           | MATH-500 、 checkpoint + eval   |
|   |                           | gradient checkpointing、、zero-adv  |

。[reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch)  Chapter 7  GRPO （Olmo3 、DeepSeek-V3.2 、GDPO ）， MATH-500 。

## RLVR 

RLVR ，：

1. ****：、、。""""""，RLVR 。， RM 。

2. ** hack**：，""。，""，。

3. **"RLVR ？"**： 2025  NeurIPS  oral 。 RLVR （），。。

---

GRPO  Critic，RLVR  RM。， RL 。 RL —— RL Scaling  Test-time Scaling。[ 12 ](../chapter12_future_trends/rl-scaling-outlook)——RL Scaling 。
