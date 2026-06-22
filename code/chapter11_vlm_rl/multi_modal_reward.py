"""
11：
==========================================================

 VLM（）：
  1. reward_correctness:       （0.0  1.0）
  2. reward_reasoning_quality: （0.0~0.5）
  3. reward_format:            （0.0~0.2）
  4. reward_visual_grounding:  （0.0~0.3）
  5. compute_total_reward:     

：
  - ，
  - ""
  - 
  - ""

：
  python multi_modal_reward.py
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

# 
os.makedirs("output", exist_ok=True)

# ，
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# ：
# ==========================================

def reward_correctness(response, ground_truth):
    """
    ：

    ：
      1. 
      2.  ground_truth 
      3.  1.0， 0.0

    ：
        response: 
        ground_truth: dict，{'': int, '': int, '': int}
    ：
        float: 0.0（） 1.0（）
    """
    # 
    extracted = {}

    for shape_name in ['', '', '']:
        # 1： + ， "3"  "：3"  "3"
        patterns = [
            rf'{shape_name}[^0-9]*?(\d+)',
            rf'(\d+)\s*\s*{shape_name}',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                # （）
                extracted[shape_name] = int(matches[-1])
                break

    # ， 0
    for shape_name in ground_truth:
        if shape_name not in extracted:
            return 0.0

    # 
    for shape_name, expected_count in ground_truth.items():
        if extracted.get(shape_name, -1) != expected_count:
            return 0.0

    return 1.0


def reward_reasoning_quality(response):
    """
    ：

    （0.0~0.5）：
      - 0.0: ，
      - 0.1~0.2: 
      - 0.3~0.4: 
      - 0.5: ，

    ：
      - （""、"Step 1"、"1." ）
      - （""、""、""、""、""）
      - 

    ：
        response: 
    ：
        float: 0.0~0.5
    """
    score = 0.0

    # （0.2 ）
    step_markers = [
        r'[\d]+',
        r'[Ss]tep\s*\d+',
        r'',
        r'',
        r'',
        r'',
    ]
    step_count = sum(1 for m in step_markers if re.search(m, response))
    if step_count >= 3:
        score += 0.2
    elif step_count >= 1:
        score += 0.1

    # （0.2 ）
    reasoning_keywords = [
        '', '', '', '', '',
        '', '', '', '', '',
    ]
    keyword_count = sum(1 for kw in reasoning_keywords if kw in response)
    if keyword_count >= 3:
        score += 0.2
    elif keyword_count >= 1:
        score += 0.1

    # （0.1 ）——
    if len(response) >= 50:
        score += 0.1

    return min(score, 0.5)


def reward_format(response):
    """
    ：

    （0.0~0.2）：
      - 0.0: 
      - 0.05: （）
      - 0.1: （）
      - 0.15: （）
      - 0.2: （ + ）

    ：
      " X 、Y  Z ， N 。"

    ：
        response: 
    ：
        float: 0.0~0.2
    """
    score = 0.0

    # （0.1 ）
    shapes_mentioned = sum(1 for s in ['', '', ''] if s in response)
    if shapes_mentioned == 3:
        score += 0.1
    elif shapes_mentioned >= 2:
        score += 0.05

    # （0.05 ）
    summary_patterns = [
        r'\s*\d+',
        r'\s*\d+',
        r'\s*\d+',
        r'\s*\d+',
        r'[：:]',
    ]
    if any(re.search(p, response) for p in summary_patterns):
        score += 0.05

    # （0.05 ）
    has_number_shape_pair = bool(re.search(r'\d+\s*', response))
    if has_number_shape_pair:
        score += 0.05

    return min(score, 0.2)


def reward_visual_grounding(response, image_info):
    """
    ：

    （0.0~0.3）：
      - 
      - ，（、）
      - ""

    image_info ：
      {
        'present_shapes': ['', ''],  # 
        'absent_shapes': [''],             # 
      }

    ：
        response: 
        image_info: dict，
    ：
        float: 0.0~0.3
    """
    score = 0.0

    present_shapes = image_info.get('present_shapes', [])
    absent_shapes = image_info.get('absent_shapes', [])

    # （0.15 ）
    mentioned_present = sum(1 for s in present_shapes if s in response)
    if len(present_shapes) > 0:
        ratio = mentioned_present / len(present_shapes)
        score += 0.15 * ratio

    # （0.1 ）
    # ，"0 """，
    for shape in absent_shapes:
        if shape in response:
            #  0
            zero_patterns = [
                rf'{shape}[^0-9]*?0',
                rf'0\s*\s*{shape}',
                rf'\s*{shape}',
            ]
            if any(re.search(p, response) for p in zero_patterns):
                score += 0.05
                break  #  0.05

    # （0.05 ）
    visual_keywords = [
        '', '', '', '', '', '',
        '', '', '', '', '',
        '', '', '',
        '', '', '', '',
    ]
    visual_count = sum(1 for kw in visual_keywords if kw in response)
    if visual_count >= 2:
        score += 0.05

    return min(score, 0.3)


def compute_total_reward(response, ground_truth, image_info):
    """
    

    ：
      - （correctness）:       × 1.0  →  1.0
      - （reasoning）:        × 1.0  →  0.5
      - （format）:           × 1.0  →  0.2
      - （visual_grounding）: × 1.0  →  0.3
      ---------------------------------------------
      : 2.0

    ：
        response: 
        ground_truth: dict，
        image_info: dict，
    ：
        dict，
    """
    r_correct = reward_correctness(response, ground_truth)
    r_reasoning = reward_reasoning_quality(response)
    r_format = reward_format(response)
    r_visual = reward_visual_grounding(response, image_info)

    total = r_correct + r_reasoning + r_format + r_visual

    return {
        'correctness': r_correct,
        'reasoning': r_reasoning,
        'format': r_format,
        'visual_grounding': r_visual,
        'total': total,
    }


# ==========================================
# ：
# ==========================================
# 8 ，：
#   、、、、
#   、、、

test_cases = [
    {
        'name': '（ +  +  + ）',
        'response': (
            '。\n'
            '，： 3 ，。\n'
            '，： 1 ，，。\n'
            '，： 2 ，。\n'
            '， 3 、1  2 ， 6 。\n'
            '：3，1，2，6。'
        ),
        'ground_truth': {'': 3, '': 1, '': 2},
        'image_info': {
            'present_shapes': ['', '', ''],
            'absent_shapes': [],
        },
    },
    {
        'name': '（）',
        'response': '3，1，2。',
        'ground_truth': {'': 3, '': 1, '': 2},
        'image_info': {
            'present_shapes': ['', '', ''],
            'absent_shapes': [],
        },
    },
    {
        'name': '',
        'response': (
            '。\n'
            '，。 2 。\n'
            '，。 1 。\n'
            '，。 2 。\n'
            '， 2+1+2=5 。\n'
            '：2，1，2。'
        ),
        'ground_truth': {'': 3, '': 1, '': 2},
        'image_info': {
            'present_shapes': ['', '', ''],
            'absent_shapes': [],
        },
    },
    {
        'name': '',
        'response': '，235。',
        'ground_truth': {'': 3, '': 1, '': 2},
        'image_info': {
            'present_shapes': ['', '', ''],
            'absent_shapes': [],
        },
    },
    {
        'name': '（）',
        'response': (
            '。\n'
            ' 3 ，。\n'
            ' 2 ， 1 。\n'
            ' 6 。'
        ),
        'ground_truth': {'': 3, '': 1, '': 2},
        'image_info': {
            'present_shapes': ['', '', ''],
            'absent_shapes': [],
        },
    },
    {
        'name': '',
        'response': (
            '：，：\n'
            '0，5，5，10。'
        ),
        'ground_truth': {'': 0, '': 2, '': 3},
        'image_info': {
            'present_shapes': ['', ''],
            'absent_shapes': [''],
        },
    },
    {
        'name': '',
        'response': (
            '，，'
            '，。\n'
            '1，2，2，5。'
        ),
        'ground_truth': {'': 2, '': 1, '': 3},
        'image_info': {
            'present_shapes': ['', '', ''],
            'absent_shapes': [],
        },
    },
    {
        'name': '（）',
        'response': (
            '，： 2 。\n'
            '，： 3 。\n'
            '，：，0。\n'
            '，2，3，0，5。\n'
            '：2，3，0。'
        ),
        'ground_truth': {'': 2, '': 3, '': 0},
        'image_info': {
            'present_shapes': ['', ''],
            'absent_shapes': [''],
        },
    },
]


# ==========================================
# ：
# ==========================================
def run_reward_tests():
    """
    ，
    """
    print("=" * 80)
    print("   — ")
    print("=" * 80)
    print()
    print("：")
    print("  （correctness）:       0.0 ~ 1.0   ")
    print("  （reasoning）:        0.0 ~ 0.5   ")
    print("  （format）:           0.0 ~ 0.2   ")
    print("  （visual_grounding）: 0.0 ~ 0.3   ")
    print("  ----------------------------------------------")
    print("  :                        0.0 ~ 2.0")
    print()

    # 
    header = (
        f"{'':>4s}  "
        f"{'':>6s}  "
        f"{'':>4s}  "
        f"{'':>4s}  "
        f"{'':>4s}  "
        f"{'':>5s}  "
        f"{''}"
    )
    separator = "-" * 80
    print(header)
    print(separator)

    all_rewards = []

    for i, tc in enumerate(test_cases):
        rewards = compute_total_reward(
            tc['response'],
            tc['ground_truth'],
            tc['image_info'],
        )
        all_rewards.append(rewards)

        print(
            f"  {i+1:>2d}  "
            f"{rewards['correctness']:>6.2f}  "
            f"{rewards['reasoning']:>4.2f}  "
            f"{rewards['format']:>4.2f}  "
            f"{rewards['visual_grounding']:>4.2f}  "
            f"{rewards['total']:>5.2f}  "
            f"{tc['name']}"
        )

    print(separator)
    print()

    # 
    totals = [r['total'] for r in all_rewards]
    print(f"：")
    print(f"  : {max(totals):.2f}")
    print(f"  : {min(totals):.2f}")
    print(f"  : {np.mean(totals):.2f}")
    print()

    # 
    print("=" * 80)
    print("  ")
    print("=" * 80)
    for i, (tc, rewards) in enumerate(zip(test_cases, all_rewards)):
        gt = tc['ground_truth']
        print(f"\n---  {i+1}: {tc['name']} ---")
        print(f"  : ={gt['']}, ={gt['']}, ={gt['']}")
        print(f"  : {tc['response'][:80]}...")
        print(f"  :")
        print(f"    :   {rewards['correctness']:.2f}  {'[]' if rewards['correctness'] == 1.0 else '[]'}")
        print(f"    : {rewards['reasoning']:.2f}  {'[]' if rewards['reasoning'] == 0.5 else ''}")
        print(f"    : {rewards['format']:.2f}  {'[]' if rewards['format'] == 0.2 else ''}")
        print(f"    : {rewards['visual_grounding']:.2f}  {'[]' if rewards['visual_grounding'] == 0.3 else ''}")
        print(f"    : {rewards['total']:.2f}")

    return all_rewards


# ==========================================
# ：
# ==========================================
def plot_reward_weights(all_rewards):
    """
    

    ：（）
    ：
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ---------- ： ----------
    ax1 = axes[0]

    # 
    max_scores = [1.0, 0.5, 0.2, 0.3]  # correctness, reasoning, format, visual
    labels = ['\n(1.0)', '\n(0.5)', '\n(0.2)', '\n(0.3)']
    colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71']
    explode = (0.05, 0.02, 0.02, 0.02)

    wedges, texts, autotexts = ax1.pie(
        max_scores,
        labels=labels,
        colors=colors,
        explode=explode,
        autopct=lambda pct: f'{pct:.1f}%',
        startangle=90,
        textprops={'fontsize': 10},
    )
    for autotext in autotexts:
        autotext.set_fontsize(10)
    ax1.set_title('', fontsize=14, fontweight='bold')

    # ---------- ： ----------
    ax2 = axes[1]

    n_cases = len(all_rewards)
    x = np.arange(n_cases)

    correctness_vals = [r['correctness'] for r in all_rewards]
    reasoning_vals = [r['reasoning'] for r in all_rewards]
    format_vals = [r['format'] for r in all_rewards]
    visual_vals = [r['visual_grounding'] for r in all_rewards]

    bar_width = 0.6
    ax2.bar(x, correctness_vals, bar_width, label='', color='#e74c3c')
    ax2.bar(x, reasoning_vals, bar_width, bottom=correctness_vals,
            label='', color='#3498db')
    bottom2 = [c + r for c, r in zip(correctness_vals, reasoning_vals)]
    ax2.bar(x, format_vals, bar_width, bottom=bottom2,
            label='', color='#f39c12')
    bottom3 = [b + f for b, f in zip(bottom2, format_vals)]
    ax2.bar(x, visual_vals, bar_width, bottom=bottom3,
            label='', color='#2ecc71')

    # 
    for i, r in enumerate(all_rewards):
        ax2.text(i, r['total'] + 0.05, f"{r['total']:.2f}",
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_xlabel('', fontsize=12)
    ax2.set_ylabel('', fontsize=12)
    ax2.set_title('', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'#{i+1}' for i in range(n_cases)])
    ax2.legend(fontsize=10, loc='upper right')
    ax2.set_ylim(0, 2.3)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('output/multi_modal_reward_breakdown.png', dpi=150, bbox_inches='tight')
    print("\n   output/multi_modal_reward_breakdown.png")
    plt.show()


# ==========================================
# 
# ==========================================
if __name__ == "__main__":
    # ：
    all_rewards = run_reward_tests()

    # ：
    print("\n" + "=" * 80)
    print("  ...")
    print("=" * 80)
    plot_reward_weights(all_rewards)

    # 
    print("\n" + "=" * 80)
    print("  ")
    print("=" * 80)
    print("""
  ：

    1. （， 1.0）
       - ， 0 
       - ""
       -  GRPO ，

    2. （ 0.5）
       - ，
       - 
       -  Chain-of-Thought 

    3. （ 0.2）
       - 
       - 
       - 

    4. （ 0.3）
       - ""
       - 
       -  VLM ，""

   RL ：
    -  RL 
    - VLM RL 
    - ，""

  ：
    -  OCR （）
    - （）
    -  LLM-as-Judge ，
    """)
