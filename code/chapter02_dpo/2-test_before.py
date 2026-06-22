import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# （ 0-download_model.py ）
LOCAL_MODEL_DIR = "./Qwen2.5-0.5B-Instruct"
model_name = LOCAL_MODEL_DIR if os.path.exists(LOCAL_MODEL_DIR) else "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

#  prompt：（ prompt ）
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
