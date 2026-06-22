"""
7：DPO  —— 
==========================================================

 DPO ：
  1.  GSM8K （）
  2. （chosen）（rejected）
  3.  DPO ，
  4. 
  5.  DPO 

：
  - ""
  - （）
  - DPO 

：
  pip install -r requirements.txt
  python dpo_math_reward.py
"""

import re
import json
import torch
import random
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

# ==========================================
# 1.  GSM8K 
# ==========================================
# ，
# ，

def create_math_preference_data():
    """
    

    ：
    - prompt: 
    - chosen: 、
    - rejected: 
    """

    math_examples = [
        {
            "prompt": " 15 ， 7 ， 3 。？",
            "chosen": "。\n： 15 。\n： 7 ， 15 - 7 = 8 。\n： 3 ， 8 + 3 = 11 。\n： 11 。",
            "rejected": " 15 。\n 7 ， 15 - 7 = 9 。（：15-7=8，9）\n 3 ， 9 + 3 = 12 。\n： 12 。",
            "answer": 11,
        },
        {
            "prompt": " 240 ， 30 ， 5 。？",
            "chosen": "。\n： 30 ， 5 。\n： 30 × 5 = 150 。\n： 240 ， 240 - 150 = 90 。\n： 90 。",
            "rejected": " 30 ， 5 。\n 30 + 5 = 35 。（：，）\n 240 - 35 = 205 。\n： 205 。",
            "answer": 90,
        },
        {
            "prompt": " 12 ， 8 ，。",
            "chosen": " =  × 。\n = 12 ， = 8 。\n = 12 × 8 = 96 。\n： 96 。",
            "rejected": " =  + 。（：，）\n = 12 + 8 = 20 。\n： 20 。",
            "answer": 96,
        },
        {
            "prompt": " 36  24 ，？",
            "chosen": " 36 ， 24 。\n 36 - 24 = 12 。\n： 12 。",
            "rejected": " 36 ， 24 。\n 36 + 24 = 60 。（：，）\n： 60 。",
            "answer": 12,
        },
        {
            "prompt": " 48 ， 6 ，？",
            "chosen": " 48 ， 6 。\n 48 ÷ 6 = 8 。\n： 8 。",
            "rejected": " 48 ， 6 。\n 48 - 6 = 42 。（：，）\n： 42 。",
            "answer": 8,
        },
        {
            "prompt": " 10 ， 6 ，。",
            "chosen": " =  ×  ÷ 2。\n = 10 ， = 6 。\n = 10 × 6 ÷ 2 = 30 。\n： 30 。",
            "rejected": " =  × 。（：2）\n = 10 × 6 = 60 。\n： 60 。",
            "answer": 30,
        },
        {
            "prompt": " 80 ， 3.5 ，？",
            "chosen": " = 80 /， = 3.5 。\n =  ×  = 80 × 3.5 = 280 。\n： 280 。",
            "rejected": " = 80 /， = 3.5 。\n = 80 + 3.5 = 83.5 。（：，）\n： 83.5 。",
            "answer": 280,
        },
        {
            "prompt": " 3 ， 12 ， 2 ， 8 。？",
            "chosen": "：3 × 12 = 36 。\n：2 × 8 = 16 。\n 36 + 16 = 52 。\n： 52 。",
            "rejected": "：3 × 12 = 36 。\n：2 × 8 = 16 。\n 36 × 16 = 576 。（：，）\n： 576 。",
            "answer": 52,
        },
        {
            "prompt": " 9 ，。",
            "chosen": " =  × 4。\n = 9 。\n = 9 × 4 = 36 。\n： 36 。",
            "rejected": " =  × 。（：，）\n = 9 × 9 = 81 。\n： 81 。",
            "answer": 36,
        },
        {
            "prompt": " 42 ， 12 。5 ？",
            "chosen": "。\n 42 - 12 = 30 。\n5  30 。\n：5  30 。",
            "rejected": "5  42 + 5 = 47 ， 12 + 5 = 17 。\n 47 - 12 = 35 。（：5，）\n：5  35 。",
            "answer": 30,
        },
    ]

    return math_examples


# ==========================================
# 2. （）
# ==========================================

def create_test_data():
    """
    ，
     DPO 
    """
    test_examples = [
        {
            "prompt": " 56 ， 8 ，？",
            "answer": 7,
        },
        {
            "prompt": " 5 ， 7 ，？",
            "answer": 35,
        },
        {
            "prompt": " 15 ， 4 ，。",
            "answer": 38,
        },
        {
            "prompt": " 100 ， 37 ， 20 ，？",
            "answer": 83,
        },
        {
            "prompt": " 24 ， 5 ，？",
            "answer": 120,
        },
    ]
    return test_examples


# ==========================================
# 3. 
# ==========================================

def extract_number(text):
    """"""
    #  "：... X"  "：X" 
    patterns = [
        r"[：:][^0-9]*?(\d+)",
        r"\s*(\d+)",
        r"\s*(\d+)",
        r"\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.findall(pattern, text)
        if match:
            return int(match[-1])  # 

    # ，
    numbers = re.findall(r"\d+", text)
    if numbers:
        return int(numbers[-1])
    return None


def evaluate_math(model, tokenizer, test_data, label=""):
    """
    

    ，，
    """
    print("=" * 60)
    print(f"【{label}】")
    print("=" * 60)

    correct = 0
    total = len(test_data)

    for i, item in enumerate(test_data):
        prompt = item["prompt"]
        true_answer = item["answer"]

        # 
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        predicted_answer = extract_number(response)

        is_correct = predicted_answer == true_answer
        correct += int(is_correct)

        status = "" if is_correct else ""
        print(f"   {i+1}: {prompt[:30]}...")
        print(f"    : {response[:80]}...")
        print(f"    : {predicted_answer} | : {true_answer} | {status}")
        print()

    accuracy = correct / total * 100
    print(f"  : {correct}/{total} = {accuracy:.1f}%")
    print("=" * 60)
    return accuracy


# ==========================================
# 4. 
# ==========================================

print("=" * 60)
print("  DPO ")
print("  ——")
print("=" * 60)
print()

# 
math_data = create_math_preference_data()
test_data = create_test_data()

print(f": {len(math_data)} ")
print(f": {len(test_data)} ")
print()
print("：")
print("  - chosen:  （）")
print("  - rejected: （）")
print("  - ，")
print()


# ==========================================
# 5. 
# ==========================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print(f" {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model_before = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

before_accuracy = evaluate_math(model_before, tokenizer, test_data, label="")

# 
del model_before
torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ==========================================
# 6. DPO 
# ==========================================

print("\n DPO ，...")

# 
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
train_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_tokenizer.pad_token = train_tokenizer.eos_token

#  HuggingFace Dataset
data_dict = {
    "prompt": [item["prompt"] for item in math_data],
    "chosen": [item["chosen"] for item in math_data],
    "rejected": [item["rejected"] for item in math_data],
}
train_dataset = Dataset.from_dict(data_dict)

# 
training_args = DPOConfig(
    output_dir="./dpo_math_results",
    per_device_train_batch_size=2,
    learning_rate=5e-5,
    num_train_epochs=5,         # 
    logging_steps=2,
    save_strategy="no",
    bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    remove_unused_columns=False,
    beta=0.1,
)

#  DPOTrainer
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=train_tokenizer,
)

print("\n...  loss  reward \n")
train_result = trainer.train()

# ==========================================
# 7. 
# ==========================================

print("\n" + "=" * 60)
print("【】")
print("=" * 60)
print(f" Loss: {train_result.training_loss:.4f}")
print()

# 
log_history = trainer.state.log_history
print("：")
print(f"{'Step':>6} | {'Loss':>8} | {'Chosen Reward':>14} | {'Rejected Reward':>16} | {'Margin':>8}")
print("-" * 70)

for entry in log_history:
    if "loss" in entry:
        step = entry.get("step", "?")
        loss = entry["loss"]
        chosen_r = entry.get("rewards/chosen", "N/A")
        rejected_r = entry.get("rewards/rejected", "N/A")
        margin = entry.get("rewards/margins", "N/A")

        # 
        if isinstance(chosen_r, float):
            chosen_r = f"{chosen_r:.4f}"
        if isinstance(rejected_r, float):
            rejected_r = f"{rejected_r:.4f}"
        if isinstance(margin, float):
            margin = f"{margin:.4f}"

        print(f"{step:>6} | {loss:>8.4f} | {chosen_r:>14} | {rejected_r:>16} | {margin:>8}")

# 
save_path = "./dpo_math_results/final_model"
trainer.save_model(save_path)
print(f"\n {save_path}")


# ==========================================
# 8. 
# ==========================================

# 
print("\n...")
model_after = AutoModelForCausalLM.from_pretrained(save_path, device_map="auto")
eval_tokenizer = AutoTokenizer.from_pretrained(save_path)
eval_tokenizer.pad_token = eval_tokenizer.eos_token

after_accuracy = evaluate_math(model_after, eval_tokenizer, test_data, label="")


# ==========================================
# 9. 
# ==========================================

print("\n" + "=" * 60)
print("【DPO  — 】")
print("=" * 60)
print()
print(f"  : {before_accuracy:.1f}%")
print(f"  : {after_accuracy:.1f}%")
print(f"  : {after_accuracy - before_accuracy:+.1f}%")
print()

print("=" * 60)
print("【】")
print("=" * 60)
print("""
1. ：
   ，""
    chosen/rejected 。。

2. DPO ：
   - DPO （）
   - ，
   -  0.5B ，

3.  RLHF ：
   -  RLHF:  →  PPO 
   - DPO : ，
   - （），DPO + 

4. ：
   - （7B、14B）
   - （ GSM8K ）
   -  Chain-of-Thought 
   -  rejected 
""")
