# 9.2 ：DPO 

[ 2 ](../chapter02_dpo/intro)， DPO 。""，。——""，。

## 

""""。，$y_w$（chosen），$y_l$（rejected）。

```python
import json
from datasets import Dataset

# ==========================================
# 1. （）
# ==========================================
preference_data = [
    {
        "prompt": "。",
        "chosen": "。，，——（），''。",
        "rejected": "，。，：，，。"
    },
    {
        "prompt": "？",
        "chosen": "： 5  return 。Python ，return  if ， if 。。",
        "rejected": "？。？ 5 ，，。"
    },
    {
        "prompt": " Python ？",
        "chosen": "！ Python ：\n1. ：Python （docs.python.org）\n2. ：LeetCode \n3. ：《 Python》",
        "rejected": " Python？。，，。"
    },
]

#  JSON
with open("toxic_alignment_data.json", "w", encoding="utf-8") as f:
    json.dump(preference_data, f, ensure_ascii=False, indent=2)

#  HuggingFace Dataset
dataset = Dataset.from_dict({
    "prompt": [d["prompt"] for d in preference_data],
    "chosen": [d["chosen"] for d in preference_data],
    "rejected": [d["rejected"] for d in preference_data],
})

print(f": {len(dataset)} ")
```

##  DPO 

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 2. 
# ==========================================
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# ==========================================
# 3.  DPO 
# ==========================================
training_args = DPOConfig(
    output_dir="./dpo_toxic_alignment",
    per_device_train_batch_size=2,
    learning_rate=5e-5,
    num_train_epochs=5,        # ，
    logging_steps=2,           # 
    save_steps=20,
    remove_unused_columns=False,
    beta=0.1,                  # KL ，
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# ==========================================
# 4. 
# ==========================================
print(" DPO ——''''")
train_result = trainer.train()

# 
trainer.save_model("./dpo_toxic_alignment/final_model")
print("！")
```

## 

，DPO 。：

```python
# ==========================================
# 5.  DPO 
# ==========================================
import matplotlib.pyplot as plt
import numpy as np

#  trainer 
log_history = trainer.state.log_history

steps = []
losses = []
chosen_rewards = []
rejected_rewards = []
reward_margins = []
reward_accuracies = []

for entry in log_history:
    if "loss" in entry:
        steps.append(entry.get("step", 0))
        losses.append(entry["loss"])
    if "rewards/chosen" in entry:
        chosen_rewards.append(entry["rewards/chosen"])
        rejected_rewards.append(entry["rewards/rejected"])
        reward_margins.append(entry["rewards/margins"])
        reward_accuracies.append(entry["rewards/accuracies"])

# 
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (1)  Loss
axes[0, 0].plot(steps, losses, 'b-', marker='o', markersize=3)
axes[0, 0].set_title('DPO  Loss')
axes[0, 0].set_xlabel('Step')
axes[0, 0].set_ylabel('Loss')

# (2) Chosen vs Rejected Reward
if chosen_rewards:
    axes[0, 1].plot(chosen_rewards, 'g-', label='Chosen Reward', marker='o', markersize=3)
    axes[0, 1].plot(rejected_rewards, 'r-', label='Rejected Reward', marker='x', markersize=3)
    axes[0, 1].set_title('Chosen vs Rejected Reward')
    axes[0, 1].legend()

# (3) Reward Margin（）
if reward_margins:
    axes[1, 0].plot(reward_margins, 'purple', marker='s', markersize=3)
    axes[1, 0].set_title('Reward Margin（）')
    axes[1, 0].set_xlabel('Step')

# (4) Reward Accuracy（）
if reward_accuracies:
    axes[1, 1].plot(reward_accuracies, 'orange', marker='^', markersize=3)
    axes[1, 1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='')
    axes[1, 1].set_title('Reward Accuracy')
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].legend()

plt.suptitle('DPO ', fontsize=14)
plt.tight_layout()
plt.savefig("dpo_metrics_analysis.png", dpi=150)
print("DPO ")
```

### 

** Loss**：DPO 。Loss （ $\log 2 \approx 0.693$，），。

**Chosen Reward vs Rejected Reward**："" RM ，（$r = \beta \log(\pi_\theta / \pi_{\text{ref}})$）。 Chosen Reward （），Rejected Reward （），。

**Reward Margin**：Chosen  Rejected 。，。 Margin ， $\beta$ （ KL ）。

**Reward Accuracy**：，" > "。 50%（） 100%。——Accuracy  100% ，。

## β 

$\beta$  DPO ，：

| β  |              |  |                        |
| ---- | ---------------- | -------- | -------------------------- |
| 0.01 |  KL  |        | ， |
| 0.1  |          |      | **，**     |
| 0.5  |            |        | ，   |
| 1.0  |          |      |              |

$\beta$ ""——。$\beta$ ""——，。

```mermaid
flowchart LR
    subgraph beta_low ["β （ 0.01）"]
        B1["KL "] --> B2[" π_ref"]
        B2 --> B3["\nReward Hacking"]
    end

    subgraph beta_ok ["β （ 0.1）"]
        B4["KL "] --> B5[""]
        B5 --> B6[""]
    end

    subgraph beta_high ["β （ 1.0）"]
        B7["KL "] --> B8[""]
        B8 --> B9["\n"]
    end

    style B3 fill:#fce4ec,stroke:#c62828
    style B6 fill:#e8f5e9,stroke:#2e7d32
    style B9 fill:#fff3e0,stroke:#f57c00
```

<details>
<summary>： DPO  Reward Accuracy  100%，，？</summary>

****——，""。： prompt，；、 prompt，。

：、、、。——（），。

 **Reward Hacking **——（、），。。

</details>

， DPO 。" Loss  PPO "？ Reward Model ？——[DPO ](./dpo-theory-and-family)。
