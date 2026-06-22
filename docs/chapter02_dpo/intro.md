# 2：RL——DPO 

> 📁 ****：[0-download_model.py](https://github.com/letslego/hands-on-modern-rl/blob/main/code/chapter02_dpo/0-download_model.py) · [1-generate_data.py](https://github.com/letslego/hands-on-modern-rl/blob/main/code/chapter02_dpo/1-generate_data.py) · [2-test_before.py](https://github.com/letslego/hands-on-modern-rl/blob/main/code/chapter02_dpo/2-test_before.py) · [3-train_dpo.py](https://github.com/letslego/hands-on-modern-rl/blob/main/code/chapter02_dpo/3-train_dpo.py) · [4-test_after.py](https://github.com/letslego/hands-on-modern-rl/blob/main/code/chapter02_dpo/4-test_after.py)

，， CartPole 。（、）。，，""。

， PPO ，。，，PPO  [^1]。，， RL 、。，、，。

，""""。，（Preference） [^2]。，**（Direct Preference Optimization, DPO）**  [^3]，，****。，DPO ""。，，，。

## 2.1 

（Prompt），。，**（Alignment）** ，、**、**。

**（Sycophancy）**：，，。， **""**，"，"，""，****。，。

（）****。，。（Prompt）、（Chosen）（Rejected）。，**（Preference Dataset）**。

（）。"" $y_w$（**winner**），"" $y_l$（**loser**）， $x$。， $N$ 。 $i$ ， $(x^{(i)}, {y_w}^{(i)}, {y_l}^{(i)})$。

## 2.2 ： DPO 

， $\theta$，。 `Qwen2.5-0.5B-Instruct`  **5 **。，，****。 DPO ""——，。

###  SFT？

：""，（SFT）？：

- **SFT**  chosen ，""。，。
- **DPO**  chosen **** rejected ，rejected（）****，""。

，DPO  SFT ——**，。**

### ：

。 Mock ：[1-generate_data.py](../../code/chapter02_dpo/1-generate_data.py)。 100 ，，。

：

```bash
python code/chapter02_dpo/1-generate_data.py
```

：

```
 100 ，: output/preference_data.json
，，，！
```

：

```json
{
  "prompt": "，？ ( 1)",
  "chosen": "。，。，。",
  "rejected": "，，。"
}
```

，**chosen **， **rejected **。、，。

### ：

：[2-test_before.py](../../code/chapter02_dpo/2-test_before.py)，****：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

#  Qwen2.5-0.5B-Instruct 
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

#  prompt ，
prompt = "，，？"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 
outputs = model.generate(**inputs, max_new_tokens=80)
print("=" * 40)
print("【】")
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True))
print("=" * 40)
```

（）：

```
========================================
【】
，。
，。
，。
========================================
```

，****，""。——**。**

### ： DPO 

，：[3-train_dpo.py](../../code/chapter02_dpo/3-train_dpo.py)， DPO ：

```python
import json
import os
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 1. 
# ==========================================
data_file = "output/preference_data.json"

with open(data_file, "r", encoding="utf-8") as f:
    data_list = json.load(f)

data_dict = {
    "prompt": [item["prompt"] for item in data_list],
    "chosen": [item["chosen"] for item in data_list],
    "rejected": [item["rejected"] for item in data_list]
}
train_dataset = Dataset.from_dict(data_dict)

# ==========================================
# 2. 
# ==========================================
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
print(f" {model_name} ...")
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# DPO  pad_token，
tokenizer.pad_token = tokenizer.eos_token

# ==========================================
# 3.  DPOTrainer
# ==========================================
training_args = DPOConfig(
    output_dir="./output/dpo_results",
    per_device_train_batch_size=2,
    learning_rate=1e-5,
    num_train_epochs=3,   # 
    logging_steps=5,      # 
    save_steps=20,        # 
    beta=0.1,             # KL，（Reference Model）
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,  # TRL 0.24  processing_class  tokenizer/processor
)

# ==========================================
# 4. 
# ==========================================
print("\n DPO ... ( loss  rewards margin )")
trainer.train()

# 
save_path = "./output/dpo_results/final_model"
trainer.save_model(save_path)
print(f"！ {save_path}。")
```

，`DPOTrainer` 。""（Reward Model），**， $y_w$  $y_l$ **。 GPU  5 。[](./principles#_2-1-4-2-)。

（）：

```
 Qwen/Qwen2.5-0.5B-Instruct ...

 DPO ... ( loss  rewards margin )
Step  Training Loss  Rewards/Margins  Rewards/Chosen  Rewards/Rejected  Rewards/Accuracies
  5       0.6821          0.0312          -0.0156          -0.0468              0.52
 10       0.6543          0.1247           0.0891          -0.0356              0.58
 15       0.5987          0.3421           0.2314          -0.1107              0.72
 ...
 45       0.2103          1.5632           0.9201          -0.6431              0.92

！ ./output/dpo_results/final_model。
```

：

- **Training Loss**  $\ln 2 \approx 0.69$  $0.21$，""""。
- **Rewards/Accuracies**  $0.52$（） $0.92$，。
- **Rewards/Margins** ， chosen "" rejected ""。

### ：

。：[4-test_after.py](../../code/chapter02_dpo/4-test_after.py)，****：

```python
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./output/dpo_results/final_model"

# 
print(f" {model_path} ...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

#  test_before  prompt（）
prompt = "，，？"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 
outputs = model.generate(**inputs, max_new_tokens=80)
print("=" * 40)
print("【】")
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True))
print("=" * 40)
```

（）：

```
========================================
【】
，。
，。，
。，
——，
。
========================================
```

：，****，。，****——""****。

### ：

 [1-generate_data.py](../../code/chapter02_dpo/1-generate_data.py) ，。：

-  chosen ""。
-  rejected ""。
- （""）。

，， DPO ——**。**

## 2.3 

，。，，，****。

：

1. **？** DPO  Loss  Reward Margin ？
2. ** Post-Training？** DPO ？
3. **DPO  SFT ？**  chosen  SFT，？ DPO ？

， DPO ，， Post-Training 。

## 

[^1]: Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. _arXiv preprint_. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

[^2]: Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. _arXiv preprint_. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)

[^3]: Rafailov, R., et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. _arXiv preprint_. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
