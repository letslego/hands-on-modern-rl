"""
8：SFT（）
==============================

 RLHF  —— （SFT）。
：
  1.  Self-Instruct -
  2.  Qwen2.5-0.5B-Instruct 
  3.  SFTTrainer 
  4. 
"""

import os
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

# 
os.makedirs("output", exist_ok=True)

# ==========================================
# 1.  Self-Instruct 
# ==========================================

def generate_sft_data():
    """
     SFT -。
    ：、、、。
     (instruction, response) 。
    """
    sft_pairs = [
        # ----  ----
        {
            "instruction": " (3 + 5) × 2 。",
            "response": "：\n：3 + 5 = 8\n 2：8 × 2 = 16\n (3 + 5) × 2 = 16。"
        },
        {
            "instruction": " 8 ， 5 ，。",
            "response": "： =  × \n： = 8 × 5 = 40 \n 40 。"
        },
        {
            "instruction": " 3/4  2/5 ，。",
            "response": "：4  5  20。\n：\n  3/4 = 15/20\n  2/5 = 8/20\n：15/20 + 8/20 = 23/20\n 23/20（ 1  3/20）。"
        },
        # ----  ----
        {
            "instruction": "。",
            "response": "，，。，。，，。"
        },
        {
            "instruction": "。",
            "response": "，，。，。，，。，，。！"
        },
        {
            "instruction": "''，100。",
            "response": " C 。。，。 Web ，。，。"
        },
        # ----  ----
        {
            "instruction": " Python ，。",
            "response": "```python\ndef is_even(n):\n    \"\"\"\"\"\"\n    return n % 2 == 0\n\n# \nprint(is_even(4))   # True\nprint(is_even(7))   # False\n```\n（%）： 2  0，。"
        },
        {
            "instruction": " Python 。",
            "response": "Python （list）（tuple）：\n\n1. ****：，；，。\n2. ****： `[1, 2, 3]`， `(1, 2, 3)`。\n3. ****：，。\n4. ****：；，、。\n\n：，。"
        },
        {
            "instruction": " Python 。",
            "response": "```python\ndef bubble_sort(arr):\n    \"\"\"：，''\"\"\"\n    n = len(arr)\n    for i in range(n):\n        swapped = False\n        for j in range(0, n - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n                swapped = True\n        # ，\n        if not swapped:\n            break\n    return arr\n\n# \nnumbers = [64, 34, 25, 12, 22, 11, 90]\nprint(\":\", numbers)\nprint(\":\", bubble_sort(numbers))\n```\n O(n²)，。"
        },
        # ----  ----
        {
            "instruction": "，。",
            "response": "。，、。，——。\n\n：，。：\n- ****：（），。\n- ****：，。\n- ****：。"
        },
        {
            "instruction": "？。",
            "response": "：\n\n1. ****（Mercury）—— 、\n2. ****（Venus）—— \n3. ****（Earth）—— ，\n4. ****（Mars）—— \n5. ****（Jupiter）—— \n6. ****（Saturn）—— \n7. ****（Uranus）—— \n8. ****（Neptune）—— ，\n\n：「」。"
        },
        {
            "instruction": " HTTP ？GET  POST ？",
            "response": "HTTP（），''。\n\nGET  POST ：\n\n|        | GET                        | POST                      |\n|-----------|----------------------------|---------------------------|\n|        |                     |                    |\n|     | URL （）               | （）           |\n|      | （ 2KB）             |                      |\n|      | （ URL ）      |                        |\n|      | （）       |                      |\n|       |                 |                   |\n\n：GET 「」，POST 「」。"
        },
        {
            "instruction": "？。",
            "response": "。\n\n**——：**\n，，……——「」。\n\n**——：**\n```python\ndef factorial(n):\n    if n == 1:        # （）\n        return 1\n    return n * factorial(n - 1)  # ，\n\nprint(factorial(5))  # 120\n```\n\n：\n1. ****：（）\n2. ****：，"
        },
        {
            "instruction": "，？",
            "response": "。\n\n****：，——「」，。\n\n****：，——。\n\n：\n1. （ ID）（ B+ ）\n2. ，\n3. ，\n\n****：，、。，。"
        },
    ]

    return sft_pairs


# ==========================================
# 2. 
# ==========================================

def format_sft_dataset(sft_pairs, tokenizer):
    """
     instruction-response 。
     chat template 。
    """
    formatted_texts = []
    for pair in sft_pairs:
        messages = [
            {"role": "user", "content": pair["instruction"]},
            {"role": "assistant", "content": pair["response"]},
        ]
        #  chat_template 
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        formatted_texts.append(text)

    return Dataset.from_dict({"text": formatted_texts})


# ==========================================
# 3. ：SFT 
# ==========================================

def main():
    print("=" * 60)
    print("8：SFT（）")
    print("=" * 60)

    # ---- 3.1  ----
    print("\n[1]  Self-Instruct ...")
    sft_pairs = generate_sft_data()
    print(f"   {len(sft_pairs)} -")
    print(f"  ：({3})、({3})、({3})、({6})")

    # 
    sample = sft_pairs[0]
    print(f"\n  ：")
    print(f"    ：{sample['instruction']}")
    print(f"    ：{sample['response'][:50]}...")

    # ---- 3.2  ----
    print("\n[2]  Qwen2.5-0.5B-Instruct ...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #  pad_token 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  #  float32  CPU
    )
    print(f"  ：{model_name}")

    # ---- 3.3  ----
    print("\n[3] ...")
    test_instructions = [
        " Python 。",
        "。",
    ]

    print("  ---  ---")
    before_responses = []
    for inst in test_instructions:
        messages = [{"role": "user", "content": inst}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        before_responses.append(response)
        print(f"  Q: {inst}")
        print(f"  A: {response[:80]}...")
        print()

    # ---- 3.4  SFT  ----
    print("[4] ...")
    train_dataset = format_sft_dataset(sft_pairs, tokenizer)
    print(f"  ：{len(train_dataset)} ")
    print(f"  ：{train_dataset[0]['text'][:100]}...")

    # ---- 3.5  SFT  ----
    print("\n[5]  SFT ...")
    print("  ：")
    print("    - per_device_train_batch_size = 2")
    print("    - learning_rate = 2e-5")
    print("    - num_train_epochs = 2")
    print("    - max_length = 512")

    sft_config = SFTConfig(
        output_dir="./output/sft_results",
        per_device_train_batch_size=2,
        learning_rate=2e-5,
        num_train_epochs=2,
        max_length=512,
        logging_steps=1,          # （）
        save_strategy="epoch",    #  epoch 
        report_to="none",         #  wandb 
        fp16=False,               # CPU  float32
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # 
    train_result = trainer.train()

    # ---- 3.6  ----
    print("\n[6] ：")
    print("  " + "-" * 40)
    print("  Step | Training Loss")
    print("  " + "-" * 40)

    # 
    if hasattr(trainer, "state") and trainer.state.log_history:
        for log_entry in trainer.state.log_history:
            if "loss" in log_entry:
                step = log_entry.get("step", "?")
                loss = log_entry["loss"]
                print(f"  {str(step).rjust(4)} | {loss:.4f}")
    else:
        print(f"  ：{train_result.training_loss:.4f}")

    print("  " + "-" * 40)

    # ---- 3.7  ----
    print("\n[7] ...")
    print("  ---  ---")

    # 
    model.eval()
    after_responses = []
    for inst in test_instructions:
        messages = [{"role": "user", "content": inst}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        after_responses.append(response)
        print(f"  Q: {inst}")
        print(f"  A: {response[:80]}...")
        print()

    # ---- 3.8  ----
    print("=" * 60)
    print("：")
    print("=" * 60)
    for i, inst in enumerate(test_instructions):
        print(f"\n  ：{inst}")
        print(f"  ：{before_responses[i][:60]}...")
        print(f"  ：{after_responses[i][:60]}...")

    # ---- 3.9  SFT  ----
    print("\n[8]  SFT ...")
    save_path = "./output/sft_results/sft_model"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"  SFT ：{save_path}")
    print("  （ / PPO ）。")

    # 
    data_save_path = "./output/sft_results/sft_training_data.json"
    with open(data_save_path, "w", encoding="utf-8") as f:
        json.dump(sft_pairs, f, ensure_ascii=False, indent=2)
    print(f"  ：{data_save_path}")

    print("\n" + "=" * 60)
    print("SFT ！")
    print(" reward_model_training.py 。")
    print("=" * 60)


if __name__ == "__main__":
    main()
