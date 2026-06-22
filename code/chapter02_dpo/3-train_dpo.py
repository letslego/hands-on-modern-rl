import json
import os
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# （ 0-download_model.py ）
LOCAL_MODEL_DIR = "./Qwen2.5-0.5B-Instruct"

# ==========================================
# 1. 
# ==========================================
data_file = "output/preference_data.json"

if not os.path.exists(data_file):
    print(f" {data_file}！ 1-generate_data.py 。")
    exit(1)

with open(data_file, "r", encoding="utf-8") as f:
    data_list = json.load(f)

#  HuggingFace Dataset 
data_dict = {
    "prompt": [item["prompt"] for item in data_list],
    "chosen": [item["chosen"] for item in data_list],
    "rejected": [item["rejected"] for item in data_list]
}
train_dataset = Dataset.from_dict(data_dict)

# ==========================================
# 2. 
# ==========================================
model_name = LOCAL_MODEL_DIR if os.path.exists(LOCAL_MODEL_DIR) else "Qwen/Qwen2.5-0.5B-Instruct"
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
    beta=0.1,             # KL （ TRL 0.24  DPOConfig ）
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
print(" 4-test_after.py ''。")
