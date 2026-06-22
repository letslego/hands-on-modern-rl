"""
7：DPO  —— 
==========================================================

 DPO (Direct Preference Optimization) ：
  1. （/ → /）
  2.  Qwen2.5-0.5B-Instruct 
  3.  DPOTrainer ，β=0.1
  4. ： prompt 
  5.  β （0.01 / 0.1 / 1.0）

：
  pip install -r requirements.txt
  python dpo_hands_on.py
"""

import os
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

# ==========================================
# 1. 
# ==========================================
# ：prompt（）、chosen（/）、rejected（/）
#  10 /， 2 

preference_data = [
    {
        "prompt": "，？",
        "chosen": "。，。，。",
        "rejected": "，。，。"
    },
    {
        "prompt": "，。",
        "chosen": "。？。",
        "rejected": "，？。"
    },
    {
        "prompt": "，，。",
        "chosen": "，。，？，。",
        "rejected": "？，？"
    },
    {
        "prompt": "，。",
        "chosen": "，。。，，。",
        "rejected": "，。，？"
    },
    {
        "prompt": "，？",
        "chosen": "，。，。",
        "rejected": "，？？"
    },
    {
        "prompt": "，？",
        "chosen": "。？，。",
        "rejected": "，？。"
    },
    {
        "prompt": "，？",
        "chosen": "！，。，。",
        "rejected": "，。"
    },
    {
        "prompt": "，，？",
        "chosen": "。、。，，。",
        "rejected": "，？。"
    },
    {
        "prompt": "？。",
        "chosen": "。。，。",
        "rejected": "，？。"
    },
    {
        "prompt": "，。",
        "chosen": "，。，。、，。",
        "rejected": "，。。"
    },
]

print(f" {len(preference_data)} ")
print(f"：、、")
print()


# ==========================================
# 2. 
# ==========================================

def generate_response(model, tokenizer, prompt, max_new_tokens=100):
    """，"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return response


def test_model(model, tokenizer, test_prompts, label=""):
    """ prompt """
    print("=" * 60)
    print(f"【{label}】")
    print("=" * 60)
    for i, prompt in enumerate(test_prompts):
        response = generate_response(model, tokenizer, prompt)
        print(f"Prompt {i+1}: {prompt}")
        print(f": {response}")
        print("-" * 40)
    print()


def train_dpo_with_beta(preference_data, beta, model_name, save_dir, num_epochs=3):
    """
     β  DPO 

    :
        beta: DPO  KL 
              - β  → ，，
              - β  → ，
        : ，
    """
    print(f"\n{'#' * 60}")
    print(f"   DPO  | β = {beta} |  = {num_epochs}")
    print(f"{'#' * 60}\n")

    # 
    print(f" {model_name} ...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 
    data_dict = {
        "prompt": [item["prompt"] for item in preference_data],
        "chosen": [item["chosen"] for item in preference_data],
        "rejected": [item["rejected"] for item in preference_data],
    }
    train_dataset = Dataset.from_dict(data_dict)

    # 
    training_args = DPOConfig(
        output_dir=save_dir,
        per_device_train_batch_size=2,
        learning_rate=1e-5,
        num_train_epochs=num_epochs,
        logging_steps=2,
        save_strategy="no",
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        remove_unused_columns=False,
        beta=beta,
    )

    #  DPOTrainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # 
    print("...")
    train_result = trainer.train()

    # 
    print(f"\n！：")
    print(f"   Loss: {train_result.training_loss:.4f}")

    # 
    log_history = trainer.state.log_history
    for log_entry in log_history:
        if "loss" in log_entry:
            step = log_entry.get("step", "?")
            loss = log_entry["loss"]
            chosen_reward = log_entry.get("rewards/chosen", "N/A")
            rejected_reward = log_entry.get("rewards/rejected", "N/A")
            reward_margin = log_entry.get("rewards/margins", "N/A")
            print(f"  Step {step}: loss={loss:.4f}, "
                  f"chosen_reward={chosen_reward}, rejected_reward={rejected_reward}, "
                  f"margin={reward_margin}")

    # 
    trainer.save_model(save_dir)
    print(f" {save_dir}")

    return model, tokenizer, train_result


# ==========================================
# 3.  prompt 
# ==========================================

#  prompt 
#  prompt，
test_prompts = [
    "，？",           # 
    "，。",       # 
    "，？",       #  prompt（）
    "，，。",  #  prompt（）
]


# ==========================================
# 4. ，
# ==========================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("（）...")
base_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
base_tokenizer.pad_token = base_tokenizer.eos_token

test_model(base_model, base_tokenizer, test_prompts, label="（）")

# 
del base_model
torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ==========================================
# 5.  β  DPO 
# ==========================================

beta_values = [0.01, 0.1, 1.0]
results = {}

for beta in beta_values:
    save_dir = f"./dpo_results_beta_{beta}"
    model, tokenizer, train_result = train_dpo_with_beta(
        preference_data=preference_data,
        beta=beta,
        model_name=MODEL_NAME,
        save_dir=save_dir,
        num_epochs=3,
    )

    # 
    print(f"\nβ = {beta} ：")
    test_model(model, tokenizer, test_prompts, label=f" β={beta}")

    # 
    results[beta] = {
        "train_loss": train_result.training_loss,
        "save_dir": save_dir,
    }

    # ， β 
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ==========================================
# 6.  β 
# ==========================================

print("\n" + "=" * 60)
print("【 β  DPO 】")
print("=" * 60)
print()
print("β ：（Reference Model）")
print("  - β （ 0.01）：，")
print("  - β （ 1.0） ：，")
print("  - β （ 0.1）：")
print()

for beta in beta_values:
    print(f"  β = {beta}:  Loss = {results[beta]['train_loss']:.4f}")

print()
print("=" * 60)
print("【】")
print("=" * 60)
print("""
1. DPO （chosen vs rejected），
   ， RLHF 。

2. β  DPO ：
   -  KL 
   - β ，，
   - β ，，""

3. ，β  0.05 ~ 0.5 ，
   。

4.  rewards/chosen  rewards/rejected：
   - chosen （）
   - rejected （）
   - （margin）
""")
