"""
9： —— RLVR 
==========================================================

 RLVR (Reinforcement Learning with Verifiable Rewards) 
，。

：
  1. check_answer_correctness  —— 
  2. check_format              —— 
  3. check_reasoning_quality   —— 
  4. compute_total_reward      —— ，

：
  python rule_based_reward.py
"""

import re


# ==========================================
# ：
# ==========================================
def check_answer_correctness(response, ground_truth):
    """
    

    ：
      1.  \\boxed{...} （LaTeX ）
      2.  boxed，"..."、"..."
      3. 、、、、

    ：
        response: 
        ground_truth: （）
    ：
        dict: {
            "score": float (0.0  1.0),
            "extracted": str (),
            "method": str (),
            "correct": bool (),
        }
    """
    extracted = None
    method = ""

    # 1： \boxed{...} 
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed_match:
        extracted = boxed_match.group(1).strip()
        method = "\\boxed{} "

    # 2："//："
    if extracted is None:
        cn_patterns = [
            r'[：:]\s*([+-]?\d+\.?\d*)',       # " 42"
            r'[：:]\s*([+-]?\d+\.?\d*)',    # " 42"
            r'[：:]\s*([+-]?\d+\.?\d*)',         # " 42"
            r'[，,]\s*(?:[])?\s*([+-]?\d+\.?\d*)',  # " 42"
        ]
        for pattern in cn_patterns:
            match = re.search(pattern, response)
            if match:
                extracted = match.group(1).strip()
                method = ""
                break

    # 3： "The answer is ..." 
    if extracted is None:
        en_patterns = [
            r'[Tt]he answer is\s*([+-]?\d+\.?\d*)',
            r'[Tt]herefore[,.]?\s*(?:the answer is\s*)?([+-]?\d+\.?\d*)',
            r'[Ss]o the answer is\s*([+-]?\d+\.?\d*)',
        ]
        for pattern in en_patterns:
            match = re.search(pattern, response)
            if match:
                extracted = match.group(1).strip()
                method = ""
                break

    # 4：（）
    if extracted is None:
        all_numbers = re.findall(r'([+-]?\d+\.?\d*)', response)
        if all_numbers:
            extracted = all_numbers[-1]
            method = "（）"

    # 
    correct = False
    if extracted is not None:
        try:
            # ，
            extracted_num = float(extracted)
            truth_num = float(ground_truth)
            correct = abs(extracted_num - truth_num) < 1e-6
        except (ValueError, TypeError):
            # ，
            correct = str(extracted).strip() == str(ground_truth).strip()

    score = 1.0 if correct else 0.0

    return {
        "score": score,
        "extracted": extracted if extracted else "（）",
        "method": method,
        "correct": correct,
    }


# ==========================================
# ：
# ==========================================
def check_format(response):
    """
    

    ：
      1. （，""、"X"、"Step"）
      2. （""、"\\boxed{}"）
      3. （）
      4. 

    ：
        response: 
    ：
        dict: {
            "score": float (0.0 ~ 1.0),
            "details": dict (),
        }
    """
    details = {}

    # 1：（0.25 ）
    step_patterns = [
        r'\s*\d',         # "1"
        r'\s*\d+\s*',     # "1"
        r'[Ss]tep\s*\d',      # "Step 1"
        r'\d+\)\s',           # "1) "
        r'|||',  # 
    ]
    has_steps = any(re.search(p, response) for p in step_patterns)
    details[""] = 0.25 if has_steps else 0.0

    # 2：（0.25 ）
    answer_patterns = [
        r'\\boxed\{',         # LaTeX 
        r'[：:]',       # 
        r'[Tt]he answer is',  # 
        r'',           # 
    ]
    has_answer = any(re.search(p, response) for p in answer_patterns)
    details[""] = 0.25 if has_answer else 0.0

    # 3：（0.25 ）
    length = len(response)
    if 20 <= length <= 2000:
        details[""] = 0.25
    elif 10 <= length < 20 or 2000 < length <= 5000:
        details[""] = 0.10
    else:
        details[""] = 0.0

    # 4：（0.25 ）
    math_patterns = [
        r'\d+\s*[+\-*/×÷]\s*\d+',   # ：3 + 5
        r'\d+\s*[=＝]\s*\d+',        # ：x = 10
        r'[（(][^)]*[)）]',          # 
        r'\\frac|\\sqrt|\\times',     # LaTeX 
    ]
    has_math = any(re.search(p, response) for p in math_patterns)
    details[""] = 0.25 if has_math else 0.0

    total_score = sum(details.values())
    return {
        "score": total_score,
        "details": details,
    }


# ==========================================
# ：
# ==========================================
def check_reasoning_quality(response):
    """
    

    ：
      1.  —— （），
      2.  —— 
      3.  —— /
      4.  —— 、

    ：
        response: 
    ：
        dict: {
            "score": float (0.0 ~ 1.0),
            "details": dict (),
        }
    """
    details = {}

    # 1：（0 ~ 0.3 ）
    # /
    sentences = re.split(r'[。.！!？?\n]', response)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
    num_steps = len(sentences)
    if num_steps >= 5:
        details[""] = 0.30
    elif num_steps >= 3:
        details[""] = 0.20
    elif num_steps >= 1:
        details[""] = 0.10
    else:
        details[""] = 0.0

    # 2：（0 ~ 0.3 ）
    #  "="  "" 
    calc_patterns = [
        r'\d+\s*[+－\-]\s*\d+\s*[=＝]\s*\d+',   # 
        r'\d+\s*[*×]\s*\d+\s*[=＝]\s*\d+',       # 
        r'\d+\s*[/÷]\s*\d+\s*[=＝]\s*[\d.]+',   # 
        r'=+\s*\d+',                              # 
    ]
    calc_count = sum(len(re.findall(p, response)) for p in calc_patterns)
    if calc_count >= 3:
        details[""] = 0.30
    elif calc_count >= 1:
        details[""] = 0.15
    else:
        details[""] = 0.0

    # 3：（0 ~ 0.2 ）
    logic_words = [
        '', '', '', '', '',
        '', '', '', '',
        '', '', '',
        '', '', '', '',
    ]
    logic_count = sum(1 for word in logic_words if word in response)
    if logic_count >= 3:
        details[""] = 0.20
    elif logic_count >= 1:
        details[""] = 0.10
    else:
        details[""] = 0.0

    # 4：（0 ~ 0.2 ）
    # 
    error_patterns = [
        r'\s*0',           # 
        r'÷\s*0',             # （）
        r'/\s*0(?!\d)',        # （）
        r'.*',         # 
        r'.*',       # 
    ]
    has_errors = any(re.search(p, response) for p in error_patterns)
    details[""] = 0.0 if has_errors else 0.20

    total_score = sum(details.values())
    return {
        "score": min(total_score, 1.0),
        "details": details,
    }


# ==========================================
# ：
# ==========================================
def compute_total_reward(response, ground_truth, weights=None):
    """
    ，

     = w1 *  + w2 *  + w3 * 

    ：
        - （w1 = 0.6）：，
        - （w2 = 0.15）：
        -   （w3 = 0.25）：

    ：
        response: 
        ground_truth: 
        weights:  {"correctness": w1, "format": w2, "reasoning": w3}
    ：
        dict: {
            "total_reward": float (0.0 ~ 1.0),
            "answer_check": dict,
            "format_check": dict,
            "reasoning_check": dict,
        }
    """
    if weights is None:
        weights = {
            "correctness": 0.6,
            "format": 0.15,
            "reasoning": 0.25,
        }

    # 
    answer_result = check_answer_correctness(response, ground_truth)
    format_result = check_format(response)
    reasoning_result = check_reasoning_quality(response)

    # 
    total_reward = (
        weights["correctness"] * answer_result["score"]
        + weights["format"] * format_result["score"]
        + weights["reasoning"] * reasoning_result["score"]
    )

    return {
        "total_reward": total_reward,
        "answer_check": answer_result,
        "format_check": format_result,
        "reasoning_check": reasoning_result,
        "weights": weights,
    }


# ==========================================
# ：
# ==========================================
def print_reward_breakdown(result, response_label=""):
    """
    

    ：
        result: compute_total_reward 
        response_label: （）
    """
    print(f"  【{response_label}】: {result['total_reward']:.4f}")
    print(f"  ├──  ({result['weights']['correctness']:.0%} ): "
          f"{result['answer_check']['score']:.2f}")
    print(f"  │   ├── : {result['answer_check']['extracted']}")
    print(f"  │   ├── : {result['answer_check']['method']}")
    print(f"  │   └── : {'' if result['answer_check']['correct'] else ''}")
    print(f"  ├──  ({result['weights']['format']:.0%} ): "
          f"{result['format_check']['score']:.2f}")
    for item, score in result['format_check']['details'].items():
        icon = "+" if score > 0 else "-"
        print(f"  │   ├── [{icon}] {item}: {score:.2f}")
    print(f"  └──  ({result['weights']['reasoning']:.0%} ): "
          f"{result['reasoning_check']['score']:.2f}")
    for item, score in result['reasoning_check']['details'].items():
        icon = "+" if score > 0 else "-"
        print(f"      ├── [{icon}] {item}: {score:.2f}")
    print()


# ==========================================
# ：
# ==========================================
def run_tests():
    """
    

    ：
      1.  —— 、、
      2.  —— 
      3.  —— 
      4.  —— ，，
      5. LaTeX  ——  \\boxed{} 
    """
    print("=" * 70)
    print("  ")
    print("=" * 70)

    # 
    test_cases = [
        {
            "label": "",
            "ground_truth": "42",
            "response": (
                "。\n"
                "1：， 15 ， 27 。\n"
                "2： = 15 + 27 = 42。\n"
                "， 42 。\n"
                "：42"
            ),
        },
        {
            "label": "",
            "ground_truth": "42",
            "response": "42",
        },
        {
            "label": "",
            "ground_truth": "42",
            "response": (
                "，。\n"
                "， 15 ， 27 。\n"
                "， 15 + 27 = 35。\n"
                "， = 35 。\n"
                "，。\n"
                "：35"
            ),
        },
        {
            "label": "",
            "ground_truth": "42",
            "response": "",
        },
        {
            "label": "LaTeX ",
            "ground_truth": "36",
            "response": (
                "， 4 × 9。\n"
                "1：4 × 9 = 36\n"
                " \\boxed{36}"
            ),
        },
    ]

    # 
    results = []
    for tc in test_cases:
        print(f"\n{'─' * 70}")
        print(f"  : {tc['label']}")
        print(f"  : {tc['ground_truth']}")
        print(f"  : {tc['response'][:80]}{'...' if len(tc['response']) > 80 else ''}")
        print(f"{'─' * 70}")

        result = compute_total_reward(tc["response"], tc["ground_truth"])
        print_reward_breakdown(result, tc["label"])
        results.append((tc["label"], result))

    # ==========================================
    # ：
    # ==========================================
    print("=" * 70)
    print("  ")
    print("=" * 70)
    print()
    print(f"  {'':<20s}  {'':>8s}  {'':>8s}  {'':>8s}  {'':>8s}")
    print(f"  {'─' * 20}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}")
    for label, result in results:
        total = result["total_reward"]
        correct = result["answer_check"]["score"]
        fmt = result["format_check"]["score"]
        reasoning = result["reasoning_check"]["score"]
        print(f"  {label:<20s}  {total:>8.4f}  {correct:>8.2f}  {fmt:>8.2f}  {reasoning:>8.2f}")

    print()
    print("=" * 70)
    print("  ")
    print("=" * 70)
    print("""
  1. （ 60%）
     - ：
     - ，
     -  RLVR 

  2. （ 15%）
     - 
     - 、、
     - ，

  3. （ 25%）
     - 
     - 
     - 

  ：
     - （60%），
     - （25%），""
     - （15%），
    """)


# ==========================================
# 
# ==========================================
if __name__ == "__main__":
    run_tests()
