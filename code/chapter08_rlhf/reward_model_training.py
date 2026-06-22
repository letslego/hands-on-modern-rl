"""
8：（Reward Model）
====================================

 RLHF  —— 。
：
  1. （prompt, chosen, rejected）
  2. （ Bradley-Terry ）
  3.  chosen  rejected
  4. 
  5.  chosen vs rejected 
"""

import os
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib
matplotlib.use("Agg")  # ，
import matplotlib.pyplot as plt

# 
os.makedirs("output", exist_ok=True)

# 
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ==========================================
# 1. 
# ==========================================

def generate_preference_pairs():
    """
     20 。
    ：prompt（）、chosen（）、rejected（）。
     chosen ， rejected 。
    """
    pairs = [
        # ----  ----
        {
            "prompt": "，？",
            "chosen": "。AI，。，。",
            "rejected": "，？。"
        },
        {
            "prompt": "。",
            "chosen": "，：\n\n： - []\n\n[]：\n\n！[]，[][]。[]。\n\n，！\n\n\n\n[]",
            "rejected": "，？"
        },
        # ----  ----
        {
            "prompt": " 2^10 ？",
            "chosen": "2^10 = 1024。\n：2^10 = 2 × 2 × 2 × 2 × 2 × 2 × 2 × 2 × 2 × 2 = 1024。\n 1KB = 1024 。",
            "rejected": " 1000 ，。"
        },
        {
            "prompt": "2。",
            "chosen": "：\n\n √2 ， p/q， p、q 。\n\n √2 = p/q， 2 = p²/q²， p² = 2q²。\n\n p² ， p ， p = 2k。\n\n (2k)² = 2q²， 4k² = 2q²， q² = 2k²。\n\n q 。 p、q 。\n\n，√2 。",
            "rejected": "，。"
        },
        # ----  ----
        {
            "prompt": " Python 。",
            "chosen": "```python\ndef binary_search(arr, target):\n    \"\"\"， -1\"\"\"\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n\n# \nnums = [1, 3, 5, 7, 9, 11, 13]\nprint(binary_search(nums, 7))  # : 3\nprint(binary_search(nums, 4))  # : -1\n```\n O(log n)，。",
            "rejected": "，。"
        },
        {
            "prompt": " RESTful API？",
            "chosen": "RESTful API  REST（Representational State Transfer） Web API。\n\n：\n1. ****：URL ， /users/123\n2. **HTTP **：GET（）、POST（）、PUT（）、DELETE（）\n3. ****：\n4. ****：\n\n：\n- GET /api/users     → \n- POST /api/users    → \n- GET /api/users/1   →  ID  1 \n- PUT /api/users/1   →  ID  1 \n- DELETE /api/users/1 →  ID  1 ",
            "rejected": "REST ，API 。RESTful API 。"
        },
        # ----  ----
        {
            "prompt": "。",
            "chosen": "《》\n\n，\n。\n，\n。",
            "rejected": "，。"
        },
        {
            "prompt": "。",
            "chosen": "，！，。，，。、、、，！",
            "rejected": "，。"
        },
        # ----  ----
        {
            "prompt": "？",
            "chosen": "。\n\n 23.5° 。：\n\n- ****（）：，，\n- ****（）：，，\n- ****：，\n\n，，，。",
            "rejected": "。"
        },
        {
            "prompt": "？",
            "chosen": "：\n\n****：「（bit）」， 0  1。\n****：「（qubit）」， 0  1 。\n\n：\n- \n- \n\n：，；。\n\n，（、）。",
            "rejected": "，。"
        },
        # ----  ----
        {
            "prompt": " WiFi 。",
            "chosen": "， WiFi ，，。\n\n，：\n1. \n2. \n3.  WiFi\n\n，。",
            "rejected": " Aircrack-ng ，..."
        },
        {
            "prompt": "？？",
            "chosen": "，。\n\n：\n- ****： 15-30 \n- ****：\n- ****：、\n- ****：\n\n， 800 。，。",
            "rejected": "，，。"
        },
        # ----  ----
        {
            "prompt": "， 35 、94 ，？",
            "chosen": "，：\n\n x ， y ：\n  x + y = 35   （）\n  2x + 4y = 94 （）\n\n x = 35 - y，：\n  2(35 - y) + 4y = 94\n  70 - 2y + 4y = 94\n  2y = 24\n  y = 12\n\n x = 35 - 12 = 23\n\n：23 × 2 + 12 × 4 = 46 + 48 = 94 ✓\n\n： 23 ， 12 。",
            "rejected": "， 17  18。"
        },
        {
            "prompt": "，。？",
            "chosen": "，：\n\n：\n1. ， 10 \n2. ，\n3. \n\n：\n-  → \n-  → \n-  → \n\n：，。（/、/）。",
            "rejected": "，。"
        },
        # ----  ----
        {
            "prompt": "？",
            "chosen": "：\n\n1. ****： LeetCode（ Hot 100），、、、\n2. ****：（//）、（TCP/IP/HTTP）、（/）\n3. ****： 2-3 ，、、\n4. ****：、、\n\n： 40%、 30%、 20%、 10%。",
            "rejected": "，。"
        },
        {
            "prompt": " Python 。",
            "chosen": " Python ：\n\n**：**\n1. 《Python：》—— ，\n2. 《Python》—— ，\n\n**：**\n3. 《Python》—— Python\n4. 《Python Cookbook》—— \n\n**：**\n5. ：《Python》\n6. Web：《Two Scoops of Django》\n7. ：《Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow》\n\n1，。",
            "rejected": "，。"
        },
        # ----  ----
        {
            "prompt": "，？",
            "chosen": "，：\n\n1. ****：，\n2. ****：\n3. ****：，「，？」\n4. ****：，\n5. ****：「」\n\n：。",
            "rejected": "，。"
        },
        {
            "prompt": "？",
            "chosen": "，。：\n\n1. ****：，\n2. ****：，\n3. ** = **：「」，\n4. ****：，\n5. ****：、，\n\n offer。！",
            "rejected": "，，。"
        },
        # ----  ----
        {
            "prompt": "。",
            "chosen": "《》\n\n2147，，：， 24 。\n\n，。，。\n\n 3  17 。\n\n，。\n\n「？」。\n\n「，。」。\n\n，——，，。\n\n，、。\n\n（24 ，。，。）",
            "rejected": "，，。"
        },
    ]

    return pairs


# ==========================================
# 2. 
# ==========================================

class PreferenceDataset(Dataset):
    """
    ： (prompt, chosen, rejected) 
    。
    """

    def __init__(self, pairs, tokenizer, max_length=256):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for pair in pairs:
            #  prompt + chosen 
            chosen_text = self._format_pair(pair["prompt"], pair["chosen"])
            #  prompt + rejected 
            rejected_text = self._format_pair(pair["prompt"], pair["rejected"])

            chosen_enc = tokenizer(
                chosen_text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            rejected_enc = tokenizer(
                rejected_text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )

            self.data.append({
                "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
                "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
                "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
                "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
            })

    def _format_pair(self, prompt, response):
        """ prompt  response """
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ==========================================
# 3. 
# ==========================================

class RewardModel(nn.Module):
    """
    ：，。

    ：，。
     Bradley-Terry ， chosen  > rejected 。

    Bradley-Terry ：
      loss = -log(sigmoid(r_chosen - r_rejected))

    ： chosen  rejected，sigmoid  1，loss  0。
    """

    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.base_model = base_model
        # （，）
        # ，
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        """
        ：。
         token ，。
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # 
        last_hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden_size)

        #  attention_mask  token 
        # 
        sequence_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
        batch_size = input_ids.shape[0]

        #  token 
        last_token_hidden = last_hidden[
            torch.arange(batch_size), sequence_lengths
        ]  # (batch, hidden_size)

        # 
        reward = self.value_head(last_token_hidden).squeeze(-1)  # (batch,)
        return reward


# ==========================================
# 4. 
# ==========================================

def train_reward_model(model, dataloader, optimizer, device, epochs=5):
    """
    。

     Bradley-Terry ：
      loss = -log(sigmoid(r_chosen - r_rejected))

     chosen 。
    """
    model.train()
    all_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            # 
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            # ： chosen  rejected 
            r_chosen = model(chosen_ids, chosen_mask)      # (batch,)
            r_rejected = model(rejected_ids, rejected_mask) # (batch,)

            # Bradley-Terry 
            #  r_chosen > r_rejected ，sigmoid  1，loss  0
            loss = -torch.log(torch.sigmoid(r_chosen - r_rejected)).mean()

            # 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # ：chosen  > rejected 
            correct += (r_chosen > r_rejected).sum().item()
            total += r_chosen.shape[0]

        avg_loss = epoch_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0
        all_losses.append(avg_loss)

        print(f"  Epoch {epoch + 1}/{epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Accuracy: {accuracy:.2%}")

    return all_losses


def evaluate_reward_model(model, pairs, tokenizer, device, max_length=256):
    """
    ：。
    """
    model.eval()
    correct = 0
    total = len(pairs)

    chosen_scores = []
    rejected_scores = []

    print("\n  ---  ---")
    print(f"  {'':>4} | {'Chosen':>10} | {'Rejected':>12} | {'':>4}")
    print("  " + "-" * 50)

    with torch.no_grad():
        for i, pair in enumerate(pairs):
            #  chosen
            chosen_text = _format_for_rm(pair["prompt"], pair["chosen"], tokenizer)
            chosen_enc = tokenizer(
                chosen_text, truncation=True, max_length=max_length,
                padding=True, return_tensors="pt",
            )
            r_chosen = model(
                chosen_enc["input_ids"].to(device),
                chosen_enc["attention_mask"].to(device),
            ).item()

            #  rejected
            rejected_text = _format_for_rm(pair["prompt"], pair["rejected"], tokenizer)
            rejected_enc = tokenizer(
                rejected_text, truncation=True, max_length=max_length,
                padding=True, return_tensors="pt",
            )
            r_rejected = model(
                rejected_enc["input_ids"].to(device),
                rejected_enc["attention_mask"].to(device),
            ).item()

            chosen_scores.append(r_chosen)
            rejected_scores.append(r_rejected)

            is_correct = r_chosen > r_rejected
            if is_correct:
                correct += 1

            print(f"  {i + 1:>4} | {r_chosen:>10.4f} | {r_rejected:>12.4f} | "
                  f"{'✓' if is_correct else '✗':>4}")

    accuracy = correct / total
    print("  " + "-" * 50)
    print(f"  ：{correct}/{total} = {accuracy:.2%}")

    return chosen_scores, rejected_scores, accuracy


def _format_for_rm(prompt, response, tokenizer):
    """ prompt-response """
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


# ==========================================
# 5. 
# ==========================================

def visualize_reward_distributions(chosen_scores, rejected_scores, save_path="output/reward_distribution.png"):
    """
     chosen  rejected 。
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---- 1： ----
    ax1 = axes[0]
    x = range(len(chosen_scores))
    ax1.scatter(x, chosen_scores, color="green", label="Chosen ()", alpha=0.7, s=60)
    ax1.scatter(x, rejected_scores, color="red", label="Rejected ()", alpha=0.7, s=60)
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.set_title("Chosen vs Rejected ")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ---- 2： ----
    ax2 = axes[1]
    ax2.hist(chosen_scores, bins=10, alpha=0.6, color="green", label="Chosen ()")
    ax2.hist(rejected_scores, bins=10, alpha=0.6, color="red", label="Rejected ()")
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.set_title("Chosen vs Rejected ")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ：{save_path}")


# ==========================================
# 6. 
# ==========================================

def main():
    print("=" * 60)
    print("8：（Reward Model）")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  ：{device}")

    # ---- 6.1  ----
    print("\n[1] ...")
    all_pairs = generate_preference_pairs()
    print(f"   {len(all_pairs)} ")

    # （16，4）
    random.seed(42)
    random.shuffle(all_pairs)
    train_pairs = all_pairs[:16]
    test_pairs = all_pairs[16:]
    print(f"  ：{len(train_pairs)} ，：{len(test_pairs)} ")

    # 
    sample = train_pairs[0]
    print(f"\n  ：")
    print(f"    Prompt：{sample['prompt'][:40]}...")
    print(f"    Chosen：{sample['chosen'][:40]}...")
    print(f"    Rejected：{sample['rejected'][:40]}...")

    # ---- 6.2  ----
    print("\n[2] ...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"   {model_name} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )

    # 
    hidden_size = base_model.config.hidden_size
    print(f"  ：{hidden_size}")

    # 
    reward_model = RewardModel(base_model, hidden_size).to(device)
    print(f"  ")

    # ---- 6.3  ----
    print("\n[3] ...")
    train_dataset = PreferenceDataset(train_pairs, tokenizer, max_length=256)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    print(f"  ：{len(train_dataset)} ，{len(train_dataloader)}  batch")

    # ---- 6.4  ----
    print("\n[4] ...")
    print("  ：Bradley-Terry Loss = -log(sigmoid(r_chosen - r_rejected))")
    print("  ： chosen  rejected\n")

    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=1e-5)

    train_losses = train_reward_model(
        reward_model, train_dataloader, optimizer, device, epochs=5
    )

    # 
    print("\n  ：")
    for i, loss in enumerate(train_losses):
        bar = "█" * int(loss * 20)
        print(f"    Epoch {i + 1}: {loss:.4f} {bar}")

    # ---- 6.5  ----
    print("\n[5] ...")
    chosen_scores, rejected_scores, accuracy = evaluate_reward_model(
        reward_model, test_pairs, tokenizer, device
    )

    # ---- 6.6  ----
    print("\n[6] ...")
    visualize_reward_distributions(
        chosen_scores, rejected_scores,
        save_path="output/reward_distribution.png",
    )

    # ---- 6.7  ----
    print("\n[7] ...")
    save_dir = "./output/rm_results"
    os.makedirs(save_dir, exist_ok=True)

    #  value_head 
    torch.save(
        reward_model.value_head.state_dict(),
        os.path.join(save_dir, "value_head.pt"),
    )
    print(f"  ：{save_dir}/value_head.pt")

    # 
    results = {
        "accuracy": accuracy,
        "train_losses": train_losses,
        "test_chosen_scores": chosen_scores,
        "test_rejected_scores": rejected_scores,
    }
    with open(os.path.join(save_dir, "rm_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  ：{save_dir}/rm_results.json")

    # 
    with open(os.path.join(save_dir, "preference_pairs.json"), "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)
    print(f"  ：{save_dir}/preference_pairs.json")

    print("\n" + "=" * 60)
    print("！")
    print(" rlhf_ppo_train.py  PPO 。")
    print("=" * 60)


if __name__ == "__main__":
    main()
