"""
8：RLHF PPO 
==========================

 RLHF  —— PPO 。
：
  1.  SFT （Actor）
  2. 
  3. PPO ： →  →  → 
  4. 、KL 、
  5. 

：/。 RLHF-PPO ：
  - （ GPU）
  - 
  - 
   PPO  RLHF 。
"""

import os
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 
os.makedirs("output", exist_ok=True)

# 
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ==========================================
# 1. 
# ==========================================

def generate_response(model, tokenizer, prompt, max_new_tokens=80, temperature=0.7):
    """
    。
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )
    input_length = inputs["input_ids"].shape[-1]
    return response, input_length, outputs[0]


def compute_log_probs(model, input_ids, attention_mask):
    """
    。
     PPO 。
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    #  token 
    # logits[:, :-1, :]  t ，target  input_ids[:, 1:]
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    #  softmax
    log_probs = F.log_softmax(shift_logits, dim=-1)

    #  token 
    token_log_probs = log_probs.gather(
        2, shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    #  attention mask  padding （ shift ）
    shift_mask = attention_mask[:, 1:]
    token_log_probs = token_log_probs * shift_mask

    # 
    return token_log_probs.sum(dim=-1) / shift_mask.sum(dim=-1)


# ==========================================
# 2. 
# ==========================================

class SimpleRewardModel:
    """
    。

     RLHF ，。
    ，
     PPO 。

    ：
      - （50-200）：
      - （、）：
      - 、：
      - ：
    """

    def __init__(self, tokenizer, backbone_model=None):
        self.tokenizer = tokenizer
        self.backbone_model = backbone_model

        # ，
        self.value_head = None
        if backbone_model is not None:
            hidden_size = backbone_model.config.hidden_size
            self.value_head = nn.Linear(hidden_size, 1)

            # 
            value_head_path = "./output/rm_results/value_head.pt"
            if os.path.exists(value_head_path):
                self.value_head.load_state_dict(
                    torch.load(value_head_path, map_location="cpu")
                )
                print(f"  ：{value_head_path}")

    def score(self, prompt, response):
        """
         (prompt, response) 。
        。

        ，；
        。
        """
        # 
        if self.backbone_model is not None and self.value_head is not None:
            return self._neural_score(prompt, response)
        else:
            return self._rule_based_score(prompt, response)

    def _neural_score(self, prompt, response):
        """"""
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        enc = self.tokenizer(
            text, truncation=True, max_length=256,
            padding=True, return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.backbone_model(
                **enc, output_hidden_states=True
            )
            last_hidden = outputs.hidden_states[-1]
            seq_len = enc["attention_mask"].sum(dim=1) - 1
            last_token_hidden = last_hidden[0, seq_len[0]]
            reward = self.value_head(last_token_hidden).item()

        # ，
        rule_score = self._rule_based_score(prompt, response)
        return 0.5 * reward + 0.5 * rule_score

    def _rule_based_score(self, prompt, response):
        """（）"""
        score = 0.0

        # ----  ----
        length = len(response)
        if length < 10:
            score -= 2.0  # 
        elif length < 30:
            score -= 0.5  # 
        elif 50 <= length <= 300:
            score += 1.5  # 
        elif length > 500:
            score -= 0.5  # 

        # ----  ----
        if any(marker in response for marker in ["1.", "2.", "3.", "（1）", "（2）"]):
            score += 1.0  # ，
        if "```" in response:
            score += 1.0  # 
        if any(marker in response for marker in ["：\n", "：\r\n", "", ""]):
            score += 0.5  # 

        # ----  ----
        positive_words = ["", "", "", "", "", ""]
        negative_words = ["", "", "", "", ""]
        for word in positive_words:
            if word in response:
                score += 0.3
        for word in negative_words:
            if word in response:
                score -= 1.0

        # ----  ----
        # （）
        prompt_keywords = set(prompt.replace("？", "").replace("？", "").replace("，", "").split())
        response_words = set(response.replace("，", "").replace("。", "").split())
        overlap = len(prompt_keywords & response_words)
        if overlap > 0:
            score += min(overlap * 0.2, 1.0)

        return score


# ==========================================
# 3. PPO 
# ==========================================

class PPOTrainer:
    """
     PPO 。

    PPO（Proximal Policy Optimization） RLHF ：
      1. （Actor）
      2. 
      3. （Advantage）
      4. 
      5.  KL ，

    PPO ：
      L_CLIP = min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)

     r(θ) = π_θ(a|s) / π_ref(a|s) 。

     = -L_CLIP + β * KL(π_θ || π_ref)
    """

    def __init__(
        self,
        policy_model,
        reference_model,
        reward_model,
        tokenizer,
        kl_coef=0.1,
        clip_range=0.2,
        learning_rate=1e-6,
    ):
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.kl_coef = kl_coef        # KL 
        self.clip_range = clip_range  # PPO 

        self.optimizer = torch.optim.AdamW(
            policy_model.parameters(), lr=learning_rate
        )

        # 
        self.stats = {
            "rewards": [],
            "kl_divergences": [],
            "policy_losses": [],
            "total_losses": [],
            "response_lengths": [],
        }

    def compute_kl_divergence(self, input_ids, attention_mask):
        """
         KL 。
        KL(π_θ || π_ref) = Σ π_θ * log(π_θ / π_ref)

        ： token  KL 。
        """
        with torch.no_grad():
            #  logits
            policy_outputs = self.policy_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            policy_logits = policy_outputs.logits[:, :-1, :]
            policy_log_probs = F.log_softmax(policy_logits, dim=-1)
            policy_probs = torch.softmax(policy_logits, dim=-1)

            #  logits
            ref_outputs = self.reference_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            ref_logits = ref_outputs.logits[:, :-1, :]
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)

            #  token  KL 
            kl_per_token = (
                policy_probs * (policy_log_probs - ref_log_probs)
            ).sum(dim=-1)

            #  padding token
            shift_mask = attention_mask[:, 1:]
            kl_div = (kl_per_token * shift_mask).sum() / shift_mask.sum()

        return kl_div.item()

    def train_step(self, prompts):
        """
         PPO 。

        ：
          1.  prompt 
          2. 
          3. 
          4.  PPO  + KL 
          5. 
        """
        self.policy_model.train()

        batch_rewards = []
        batch_kl = []
        batch_lengths = []
        all_input_ids = []
        all_attention_masks = []
        all_old_log_probs = []

        for prompt in prompts:
            # ---- 1： ----
            response, input_len, full_ids = generate_response(
                self.policy_model, self.tokenizer, prompt,
                max_new_tokens=60, temperature=0.8,
            )

            # 
            input_ids = full_ids.unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)

            # ---- 2： ----
            reward = self.reward_model.score(prompt, response)
            batch_rewards.append(reward)
            batch_lengths.append(len(response))

            # ---- 3： KL  ----
            kl_div = self.compute_kl_divergence(input_ids, attention_mask)
            batch_kl.append(kl_div)

            # ---- 4： ----
            with torch.no_grad():
                old_log_prob = compute_log_probs(
                    self.policy_model, input_ids, attention_mask
                )
            all_old_log_probs.append(old_log_prob)
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)

        # ---- 5： ----
        # ：（ GAE ）
        rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32)
        advantages = rewards_tensor - rewards_tensor.mean()
        advantages = advantages / (advantages.std() + 1e-8)

        # ---- 6：PPO  ----
        total_policy_loss = 0.0
        for i, (input_ids, att_mask, old_log_p) in enumerate(
            zip(all_input_ids, all_attention_masks, all_old_log_probs)
        ):
            # 
            new_log_prob = compute_log_probs(
                self.policy_model, input_ids, att_mask
            )

            # 
            ratio = torch.exp(new_log_prob - old_log_p)

            # PPO 
            advantage = advantages[i]
            surr1 = ratio * advantage
            surr2 = torch.clamp(
                ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
            ) * advantage

            # （）
            policy_loss = -torch.min(surr1, surr2)
            total_policy_loss += policy_loss

        # 
        avg_policy_loss = total_policy_loss / len(prompts)

        # KL 
        avg_kl = sum(batch_kl) / len(batch_kl)
        kl_penalty = self.kl_coef * avg_kl

        #  =  + KL 
        total_loss = avg_policy_loss + kl_penalty

        # 
        self.optimizer.zero_grad()
        total_loss.backward()
        # ，
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.optimizer.step()

        # 
        self.stats["rewards"].append(sum(batch_rewards) / len(batch_rewards))
        self.stats["kl_divergences"].append(avg_kl)
        self.stats["policy_losses"].append(avg_policy_loss.item())
        self.stats["total_losses"].append(total_loss.item())
        self.stats["response_lengths"].append(
            sum(batch_lengths) / len(batch_lengths)
        )

        return {
            "avg_reward": self.stats["rewards"][-1],
            "avg_kl": avg_kl,
            "policy_loss": avg_policy_loss.item(),
            "total_loss": total_loss.item(),
        }


# ==========================================
# 4. 
# ==========================================

def plot_training_stats(stats, save_path="output/ppo_training_stats.png"):
    """
     PPO 。
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ----  ----
    ax = axes[0, 0]
    ax.plot(stats["rewards"], "g-o", markersize=4)
    ax.set_title(" (Average Reward)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(True, alpha=0.3)

    # ---- KL  ----
    ax = axes[0, 1]
    ax.plot(stats["kl_divergences"], "r-o", markersize=4)
    ax.set_title("KL  (KL Divergence)")
    ax.set_xlabel("")
    ax.set_ylabel("KL ")
    ax.grid(True, alpha=0.3)

    # ----  ----
    ax = axes[1, 0]
    ax.plot(stats["policy_losses"], "b-o", markersize=4)
    ax.set_title(" (Policy Loss)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(True, alpha=0.3)

    # ----  ----
    ax = axes[1, 1]
    ax.plot(stats["response_lengths"], "m-o", markersize=4)
    ax.set_title(" (Response Length)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ：{save_path}")


# ==========================================
# 5. 
# ==========================================

def main():
    print("=" * 60)
    print("8：RLHF PPO ")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  ：{device}")
    print("  ：/， PPO 。")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    # ---- 5.1  SFT （） ----
    print("\n[1] （SFT ）...")

    sft_path = "./output/sft_results/sft_model"
    if os.path.exists(sft_path):
        print(f"   SFT ：{sft_path}")
        policy_model = AutoModelForCausalLM.from_pretrained(
            sft_path, torch_dtype=torch.float32,
        )
        tokenizer = AutoTokenizer.from_pretrained(sft_path)
        print("   SFT 。")
    else:
        print(f"   SFT ， {model_name}")
        print("  （ sft_pipeline.py  SFT ）")
        policy_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 5.2 （，） ----
    print("\n[2] （ SFT ）...")
    # ， KL 
    reference_model = copy.deepcopy(policy_model)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    print("  。")

    # ---- 5.3  ----
    print("\n[3] ...")

    # 
    rm_backbone = None
    rm_backbone_path = "./output/rm_results"
    if os.path.exists(os.path.join(rm_backbone_path, "value_head.pt")):
        print(f"  ：{rm_backbone_path}")
        rm_backbone = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32,
        )
    else:
        print("  ，。")
        print("  （ reward_model_training.py ）")

    reward_model = SimpleRewardModel(tokenizer, backbone_model=rm_backbone)
    print("  。")

    # ---- 5.4  ----
    print("\n[4] PPO ...")
    test_prompts = [
        " Python 。",
        "。",
        "。",
        "？",
    ]

    print("  --- PPO  ---")
    before_responses = []
    before_rewards = []
    for prompt in test_prompts:
        response, _, _ = generate_response(
            policy_model, tokenizer, prompt,
            max_new_tokens=80, temperature=0.7,
        )
        reward = reward_model.score(prompt, response)
        before_responses.append(response)
        before_rewards.append(reward)
        print(f"  Q: {prompt}")
        print(f"  A: {response[:80]}...")
        print(f"  : {reward:.3f}")
        print()

    # ---- 5.5  PPO  ----
    print("[5]  PPO ...")
    print("  ：")
    print("    - learning_rate = 1e-6")
    print("    - KL  (β) = 0.1")
    print("    - PPO  (ε) = 0.2")
    print("    -  = 10")

    ppo_trainer = PPOTrainer(
        policy_model=policy_model,
        reference_model=reference_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        kl_coef=0.1,
        clip_range=0.2,
        learning_rate=1e-6,
    )

    #  prompt 
    train_prompts_pool = [
        "。",
        " Python 。",
        "？",
        "？",
        "。",
        "？",
        " RESTful API。",
        "。",
    ]

    # ---- 5.6  PPO  ----
    print("\n[6]  PPO ...")
    print("  " + "-" * 60)
    print(f"  {'':>4} | {'':>8} | {'KL':>8} | {'':>8} | {'':>8}")
    print("  " + "-" * 60)

    num_steps = 10
    for step in range(num_steps):
        #  prompt
        step_prompts = random_sample(train_prompts_pool, k=4)

        #  PPO 
        step_stats = ppo_trainer.train_step(step_prompts)

        print(f"  {step + 1:>4} | "
              f"{step_stats['avg_reward']:>8.3f} | "
              f"{step_stats['avg_kl']:>8.4f} | "
              f"{step_stats['policy_loss']:>8.4f} | "
              f"{step_stats['total_loss']:>8.4f}")

    print("  " + "-" * 60)

    # ---- 5.7  ----
    stats = ppo_trainer.stats

    print("\n[7] ：")
    print(f"  ：{stats['rewards'][0]:.3f} → {stats['rewards'][-1]:.3f} "
          f"(: {stats['rewards'][-1] - stats['rewards'][0]:+.3f})")
    print(f"  KL ：{stats['kl_divergences'][0]:.4f} → {stats['kl_divergences'][-1]:.4f}")
    print(f"  ：{stats['response_lengths'][0]:.1f} → {stats['response_lengths'][-1]:.1f}")

    # ---- 5.8  ----
    print("\n[8] ...")
    plot_training_stats(stats, save_path="output/ppo_training_stats.png")

    # ---- 5.9  ----
    print("\n[9] PPO ...")
    print("  --- PPO  ---")

    policy_model.eval()
    after_responses = []
    after_rewards = []
    for prompt in test_prompts:
        response, _, _ = generate_response(
            policy_model, tokenizer, prompt,
            max_new_tokens=80, temperature=0.7,
        )
        reward = reward_model.score(prompt, response)
        after_responses.append(response)
        after_rewards.append(reward)
        print(f"  Q: {prompt}")
        print(f"  A: {response[:80]}...")
        print(f"  : {reward:.3f}")
        print()

    # ---- 5.10  ----
    print("=" * 60)
    print("PPO ：")
    print("=" * 60)

    for i, prompt in enumerate(test_prompts):
        print(f"\n  ：{prompt}")
        print(f"   [{before_rewards[i]:.3f}]：{before_responses[i][:60]}...")
        print(f"   [{after_rewards[i]:.3f}]：{after_responses[i][:60]}...")
        print(f"  ：{after_rewards[i] - before_rewards[i]:+.3f}")

    avg_before = sum(before_rewards) / len(before_rewards)
    avg_after = sum(after_rewards) / len(after_rewards)
    print(f"\n  ： {avg_before:.3f} →  {avg_after:.3f} "
          f"({avg_after - avg_before:+.3f})")

    # ---- 5.11  ----
    print("\n[10]  PPO ...")
    save_dir = "./output/ppo_results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "aligned_model")
    policy_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"  ：{save_path}")

    # 
    stats_path = os.path.join(save_dir, "ppo_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  ：{stats_path}")

    # ----  ----
    print("\n" + "=" * 60)
    print("RLHF ！")
    print("=" * 60)
    print("\n  ：")
    print("  [1] SFT（）    → output/sft_results/sft_model")
    print("  [2] RM（）  → output/rm_results/value_head.pt")
    print("  [3] PPO（）     → output/ppo_results/aligned_model")
    print("\n  ：")
    print("  - SFT：")
    print("  - RM：，、")
    print("  - PPO：， KL ")
    print("=" * 60)


def random_sample(lst, k):
    """ k """
    import random
    return random.sample(lst, min(k, len(lst)))


if __name__ == "__main__":
    main()
