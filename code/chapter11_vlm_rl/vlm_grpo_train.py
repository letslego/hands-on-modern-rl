"""
11：VLM GRPO 
==========================================================

 VLM（） GRPO ：
  1. 
  2.  GRPO （ →  →  → ）
  3.  VLM GRPO  GRPO 
  4. ：、、
  5. 

：
  ****， VLM 。
   VLM GRPO ：
    - GPU  >= 40GB（ A100）
    - transformers  VLM （ Qwen2-VL、LLaVA）
    -  +  token 
    - （ DeepSpeed）

   VLM GRPO 。

：
  python vlm_grpo_train.py
"""

import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt

# 
os.makedirs("output", exist_ok=True)

# ，
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# ：VLM GRPO  GRPO 
# ==========================================
# VLM GRPO ：
#   1.  → （Vision Encoder）
#   2.  token →  token  LLM
#   3. 
#   4. 

def print_vlm_grpo_overview():
    """
     VLM GRPO  GRPO 
    """
    print("=" * 70)
    print("  VLM GRPO vs  GRPO — ")
    print("=" * 70)
    print()
    print("   GRPO ：")
    print("  ┌──────────┐    ┌──────────┐    ┌──────────┐")
    print("  │   │ → │ LLM   │ → │   │")
    print("  │ (prompt) │    │ (group)  │    │ (score)  │")
    print("  └──────────┘    └──────────┘    └──────────┘")
    print()
    print("  VLM GRPO ：")
    print("  ┌──────┐ ┌──────────┐    ┌──────────────┐    ┌──────────────┐")
    print("  │  │→│   │→│ VLM       │→│     │")
    print("  │(img) │ │(Encoder) │    │(vision+text) │    │(correct+vis) │")
    print("  └──────┘ └──────────┘    └──────────────┘    └──────────────┘")
    print("  ┌──────────┐     ↑")
    print("  │   │ ────┘")
    print("  │ (prompt) │")
    print("  └──────────┘")
    print()
    print("  ：")
    print("    1. ： prompt → (,  prompt) ")
    print("    2. ：LLM → VLM（LLM + Vision Encoder + ）")
    print("    3. Token： token →  token +  token")
    print("    4. ： →  + ")
    print("    5. ： → （ + ）")
    print()


# ==========================================
# ：
# ==========================================
#  VLM  GPU ，
# 

# 
STANDARD_PROMPT = "、"

# 
# ， ground_truth 
def generate_correct_response(gt):
    """（）"""
    shapes_desc = []
    for shape, count in gt.items():
        if count > 0:
            shapes_desc.append(f"{shape}{count}")
        else:
            shapes_desc.append(f"{shape}0（{shape}）")
    return (
        f"。\n"
        f"，：{gt['']}。\n"
        f"，：{gt['']}。\n"
        f"，：{gt['']}。\n"
        f"，{shapes_desc[0]}，{shapes_desc[1]}，{shapes_desc[2]}。\n"
        f"：{gt['']}，{gt['']}，{gt['']}。"
    )


def generate_short_correct_response(gt):
    """"""
    return f"{gt['']}，{gt['']}，{gt['']}。"


def generate_wrong_response(gt):
    """（）"""
    wrong_gt = dict(gt)
    shape_to_modify = random.choice(list(gt.keys()))
    #  1~2
    delta = random.choice([-2, -1, 1, 2])
    wrong_gt[shape_to_modify] = max(0, wrong_gt[shape_to_modify] + delta)
    return (
        f"。\n"
        f"{wrong_gt['']}，{wrong_gt['']}，{wrong_gt['']}。\n"
        f"{sum(wrong_gt.values())}。"
    )


def generate_partially_correct_response(gt):
    """（）"""
    shapes = list(gt.keys())
    # ，
    correct_shape = random.choice(shapes)
    wrong_gt = {}
    for s in shapes:
        if s == correct_shape:
            wrong_gt[s] = gt[s]
        else:
            wrong_gt[s] = max(0, gt[s] + random.choice([-1, 1]))
    return f"，{wrong_gt['']}，{wrong_gt['']}，{wrong_gt['']}。"


def generate_low_quality_response(gt):
    """（、）"""
    shapes = list(gt.keys())
    random.shuffle(shapes)
    nums = [max(0, gt[s] + random.choice([-2, -1, 0, 1, 2])) for s in shapes]
    return f"{shapes[0]}{nums[0]}{shapes[1]}{nums[1]}{shapes[2]}{nums[2]}"


# ，
RESPONSE_GENERATORS = [
    generate_correct_response,           # ：
    generate_short_correct_response,     # ：（）
    generate_wrong_response,             # ：（）
    generate_partially_correct_response, # ：（）
    generate_low_quality_response,       # ：
]


# ==========================================
# ：（， multi_modal_reward.py ）
# ==========================================
def extract_numbers(response, shape_names):
    """"""
    import re
    extracted = {}
    for shape_name in shape_names:
        patterns = [
            rf'{shape_name}[^0-9]*?(\d+)',
            rf'(\d+)\s*\s*{shape_name}',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                extracted[shape_name] = int(matches[-1])
                break
    return extracted


def simple_reward(response, ground_truth):
    """
    ：

    ：
      - （0.0  1.0）：
      - （0.0~0.3）：
      - （0.0~0.2）：

    ：0.0 ~ 1.5
    """
    import re

    score = 0.0

    # 
    extracted = extract_numbers(response, ['', '', ''])
    all_correct = True
    for shape, expected in ground_truth.items():
        if extracted.get(shape, -1) != expected:
            all_correct = False
            break
    if all_correct and len(extracted) == len(ground_truth):
        score += 1.0

    # 
    step_keywords = ['', '', '', '', '', '', '', '']
    step_count = sum(1 for kw in step_keywords if kw in response)
    score += min(step_count * 0.06, 0.3)

    # 
    if all(s in response for s in ['', '', '']):
        score += 0.1
    if re.search(r'[：:]', response):
        score += 0.1

    return score


# ==========================================
# ：GRPO 
# ==========================================
def create_training_samples(num_samples=20, seed=42):
    """
    

    ：
      - （）
      - 
      - ground_truth（）

    ：
        num_samples: 
        seed: 
    ：
        list[dict]: 
    """
    random.seed(seed)
    samples = []

    for i in range(num_samples):
        # （0~5）
        gt = {
            '': random.randint(0, 5),
            '': random.randint(0, 5),
            '': random.randint(0, 5),
        }

        samples.append({
            'sample_id': i,
            'prompt': STANDARD_PROMPT,
            'ground_truth': gt,
        })

    return samples


# ==========================================
# ： GRPO 
# ==========================================
def simulate_grpo_training(samples, group_size=4, num_epochs=5,
                           initial_quality=0.3, seed=42):
    """
     VLM GRPO 

    GRPO （ epoch）：
      1. ， group_size 
      2. 
      3. （advantage）
      4. （）
      5. ，

    ：
        samples: 
        group_size: （GRPO  4~16）
        num_epochs: 
        initial_quality: （0~1，）
        seed: 
    ：
        dict: 
    """
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("  VLM GRPO ")
    print("=" * 70)
    print()
    print(f"：")
    print(f"  : {len(samples)}")
    print(f"   (group_size): {group_size}")
    print(f"  : {num_epochs}")
    print(f"  : {initial_quality:.1f}")
    print()
    print("：， VLM GRPO ：")
    print("  - VLM （ Qwen2-VL、LLaVA）")
    print("  -  token ")
    print("  - GPU  >= 40GB")
    print("  - DeepSpeed  FSDP ")
    print()

    # 
    history = {
        'epoch': [],
        'accuracy': [],         # 
        'avg_reward': [],       # 
        'best_reward': [],      # 
        'avg_advantage_std': [],# （）
    }

    # ： epoch ，
    # quality_factor  initial_quality  1.0
    for epoch in range(num_epochs):
        quality_factor = initial_quality + (1.0 - initial_quality) * (epoch / max(num_epochs - 1, 1))

        epoch_correct = 0
        epoch_rewards = []
        epoch_best_rewards = []
        epoch_adv_stds = []

        for sample in samples:
            gt = sample['ground_truth']

            #  group_size 
            group_rewards = []
            for g in range(group_size):
                # 
                # ，
                if random.random() < quality_factor:
                    # 
                    if random.random() < 0.7:
                        response = generate_correct_response(gt)
                    else:
                        response = generate_short_correct_response(gt)
                else:
                    # 
                    choice = random.random()
                    if choice < 0.4:
                        response = generate_wrong_response(gt)
                    elif choice < 0.7:
                        response = generate_partially_correct_response(gt)
                    else:
                        response = generate_low_quality_response(gt)

                # 
                reward = simple_reward(response, gt)
                group_rewards.append(reward)

            # GRPO ：
            rewards_arr = np.array(group_rewards)
            mean_r = rewards_arr.mean()
            std_r = rewards_arr.std() + 1e-8
            advantages = (rewards_arr - mean_r) / std_r

            # 
            epoch_rewards.extend(group_rewards)
            epoch_best_rewards.append(max(group_rewards))
            epoch_adv_stds.append(std_r)

            # 
            best_idx = np.argmax(rewards_arr)
            if rewards_arr[best_idx] >= 1.0:
                epoch_correct += 1

        #  epoch 
        accuracy = epoch_correct / len(samples)
        avg_reward = np.mean(epoch_rewards)
        best_reward = np.mean(epoch_best_rewards)
        avg_adv_std = np.mean(epoch_adv_stds)

        history['epoch'].append(epoch + 1)
        history['accuracy'].append(accuracy)
        history['avg_reward'].append(avg_reward)
        history['best_reward'].append(best_reward)
        history['avg_advantage_std'].append(avg_adv_std)

        #  epoch 
        print(f"  Epoch {epoch+1}/{num_epochs} | "
              f": {accuracy:.3f} | "
              f": {avg_reward:.3f} | "
              f": {best_reward:.3f} | "
              f"std: {avg_adv_std:.3f}")

    return history


# ==========================================
# ：
# ==========================================
def print_before_after_comparison(samples, history, seed=42):
    """
    

    ：
      1. 
      2. 
    """
    random.seed(seed)

    print("\n" + "=" * 70)
    print("  ")
    print("=" * 70)

    #  5 
    display_samples = samples[:5]

    print("\n--- （） ---")
    for i, sample in enumerate(display_samples):
        gt = sample['ground_truth']
        # ：
        response = generate_wrong_response(gt)
        reward = simple_reward(response, gt)
        extracted = extract_numbers(response, ['', '', ''])
        is_correct = all(extracted.get(s, -1) == gt[s] for s in gt)
        print(f"\n   {i+1} (GT: ={gt['']}, ={gt['']}, "
              f"={gt['']})")
        print(f"    : {response[:60]}...")
        print(f"    : {reward:.2f} | {'' if is_correct else ''}")

    print("\n\n--- （） ---")
    for i, sample in enumerate(display_samples):
        gt = sample['ground_truth']
        # ：
        response = generate_correct_response(gt)
        reward = simple_reward(response, gt)
        extracted = extract_numbers(response, ['', '', ''])
        is_correct = all(extracted.get(s, -1) == gt[s] for s in gt)
        print(f"\n   {i+1} (GT: ={gt['']}, ={gt['']}, "
              f"={gt['']})")
        print(f"    : {response[:60]}...")
        print(f"    : {reward:.2f} | {'' if is_correct else ''}")

    # 
    print("\n" + "-" * 70)
    print("  ：")
    print(f"  {'':>12s}  {'':>10s}  {'':>10s}  {'':>10s}")
    print(f"  {'----':>12s}  {'------':>10s}  {'------':>10s}  {'----':>10s}")

    before_acc = history['accuracy'][0]
    after_acc = history['accuracy'][-1]
    before_reward = history['avg_reward'][0]
    after_reward = history['avg_reward'][-1]
    before_best = history['best_reward'][0]
    after_best = history['best_reward'][-1]

    print(f"  {'':>12s}  {before_acc:>10.3f}  {after_acc:>10.3f}  {after_acc - before_acc:>+10.3f}")
    print(f"  {'':>12s}  {before_reward:>10.3f}  {after_reward:>10.3f}  {after_reward - before_reward:>+10.3f}")
    print(f"  {'':>12s}  {before_best:>10.3f}  {after_best:>10.3f}  {after_best - before_best:>+10.3f}")


# ==========================================
# ：
# ==========================================
def plot_training_curves(history):
    """
     VLM GRPO 

     4 ：
      1. 
      2. 
      3. 
      4. （）
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("VLM GRPO （）", fontsize=16, fontweight='bold')

    epochs = history['epoch']

    # 1：
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['accuracy'], 'o-', color='#2196F3', linewidth=2,
             markersize=8, label='')
    ax1.fill_between(epochs, 0, history['accuracy'], alpha=0.1, color='#2196F3')
    ax1.set_title('', fontsize=13)
    ax1.set_xlabel(' (Epoch)')
    ax1.set_ylabel('')
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2：
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['avg_reward'], 's-', color='#FF9800', linewidth=2,
             markersize=8, label='')
    ax2.fill_between(epochs, 0, history['avg_reward'], alpha=0.1, color='#FF9800')
    ax2.set_title('', fontsize=13)
    ax2.set_xlabel(' (Epoch)')
    ax2.set_ylabel('')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3：
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['best_reward'], 'D-', color='#4CAF50', linewidth=2,
             markersize=8, label='')
    ax3.set_title('', fontsize=13)
    ax3.set_xlabel(' (Epoch)')
    ax3.set_ylabel('')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4：
    ax4 = axes[1, 1]
    ax4.plot(epochs, history['avg_advantage_std'], '^-', color='#9C27B0', linewidth=2,
             markersize=8, label='')
    ax4.set_title('（）', fontsize=13)
    ax4.set_xlabel(' (Epoch)')
    ax4.set_ylabel('')
    ax4.annotate('std  = \n（）',
                 xy=(epochs[-1] * 0.5, max(history['avg_advantage_std']) * 0.8),
                 fontsize=9, color='gray', style='italic')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/vlm_grpo_training_curves.png', dpi=150, bbox_inches='tight')
    print("\n   output/vlm_grpo_training_curves.png")
    plt.show()


# ==========================================
# ：GRPO 
# ==========================================
def demonstrate_grpo_normalization():
    """
     GRPO 

    ：，VLM  4 ，
    GRPO ""
    """
    print("\n" + "=" * 70)
    print("  GRPO （VLM ）")
    print("=" * 70)

    #  ground truth
    gt = {'': 3, '': 1, '': 2}

    print(f"\n：={gt['']}，={gt['']}，={gt['']}")
    print(f"：{STANDARD_PROMPT}")
    print()

    #  4 
    responses = [
        ("+", generate_correct_response(gt)),
        ("", generate_short_correct_response(gt)),
        ("", generate_wrong_response(gt)),
        ("", generate_low_quality_response(gt)),
    ]

    rewards = []
    print("：")
    print("-" * 70)
    for i, (desc, resp) in enumerate(responses):
        r = simple_reward(resp, gt)
        rewards.append(r)
        print(f"   {i+1} ({desc})")
        print(f"    : {resp[:70]}...")
        print(f"    : {r:.3f}")
        print()

    # GRPO 
    rewards_arr = np.array(rewards)
    mean_r = rewards_arr.mean()
    std_r = rewards_arr.std() + 1e-8
    advantages = (rewards_arr - mean_r) / std_r

    print("-" * 70)
    print("GRPO ：")
    print(f"  : {mean_r:.4f}")
    print(f"  : {std_r:.4f}")
    print()
    print(f"  {'':>4s}  {'':>8s}  {'GRPO':>10s}  {''}")
    print(f"  {'----':>4s}  {'--------':>8s}  {'----------':>10s}  {'----'}")
    for i in range(len(responses)):
        adv = advantages[i]
        if adv > 0.5:
            meaning = "，"
        elif adv > 0:
            meaning = "，"
        elif adv > -0.5:
            meaning = "，"
        else:
            meaning = "，"
        print(f"  {i+1:>4d}  {rewards[i]:>8.4f}  {adv:>+10.4f}  {meaning}")

    print()
    print("  ：GRPO ，！")
    print("   GRPO  Critic 。")


# ==========================================
# 
# ==========================================
if __name__ == "__main__":
    # ： VLM GRPO 
    print_vlm_grpo_overview()

    # ：GRPO 
    demonstrate_grpo_normalization()

    # ：
    print("\n" + "=" * 70)
    print("  ")
    print("=" * 70)
    samples = create_training_samples(num_samples=20, seed=42)
    print(f"   {len(samples)} ")

    #  ground truth
    print("\n  ：")
    for i, s in enumerate(samples[:5]):
        gt = s['ground_truth']
        print(f"     {i+1}: ={gt['']}, ={gt['']}, ={gt['']}, "
              f"={sum(gt.values())}")

    # ： GRPO 
    print()
    history = simulate_grpo_training(
        samples,
        group_size=4,
        num_epochs=5,
        initial_quality=0.3,
        seed=42,
    )

    # ：
    print_before_after_comparison(samples, history, seed=42)

    # ：
    print("\n" + "=" * 70)
    print("  ...")
    print("=" * 70)
    plot_training_curves(history)

    # 
    print("\n" + "=" * 70)
    print("  VLM GRPO ")
    print("=" * 70)
    print("""
   VLM GRPO ，：

  1. VLM GRPO  GRPO ：
     -  (, ) 
     -  token
     -  token  token  LLM
     - 

  2. GRPO ：
     - （group）
     - ，
     - ，
     -  Critic 

  3.  VLM GRPO ：
     -  transformers  VLM 
       : Qwen2VLForConditionalGeneration
     -  Vision Transformer 
     -  LLM 
     - （ token ）
     -  DeepSpeed ZeRO-3  FSDP 

  4. VLM GRPO ：
     - （VQA）
     - （Image Captioning）
     - （）
     - （--）

  5. ：
     - group_size: （ 4~16）
     - clip_ratio: PPO （ 0.2）
     - learning_rate: （ 1e-6 ~ 5e-6）
     -KL_coefficient: KL （ 0.01~0.1）
    """)
