import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./output/dpo_results/final_model"

if not os.path.exists(model_path):
    print(f" {model_path}！ 3-train_dpo.py 。")
    exit(1)

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
