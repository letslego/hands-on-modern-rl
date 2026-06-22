# 8.6 ：RLHF 

## 

****

-  base、SFT、RLHF ，。
-  benchmark、、。
-  reward hacking、、、judge 。

****

$$
\text{win rate}
= \frac{N_{win}+0.5N_{tie}}{N_{win}+N_{lose}+N_{tie}}
\quad \text{（：tie ）}
$$

$$
\Delta_{regression}
= \text{score}_{RLHF}-\text{score}_{SFT}
\quad \text{（：RLHF  SFT ）}
$$

$$
\rho_{reward,length}
= \mathrm{corr}(r_{RM}(x,y),\ |y|)
\quad \text{（：）}
$$

> ****
>
> RLHF  reward ，“、、RM ”。

RLHF ，“reward ”，“ reward”。，、。 PPO ，。

： **base model、SFT model、RLHF model** ， RLHF ，。

## 

 RLHF ，，。

|            |                          |                                 |
| -------------- | ------------------------------ | --------------------------------------- |
|  benchmark | 、、   | RLHF 、、？ |
|        |              | RLHF  SFT 、？  |
|        | reward hacking、、 | 、、？    |

。Benchmark ，“”；， judge ；，。

，：

```text
 checkpoint
  ->  benchmark：
  ->  pairwise： SFT 
  -> ： reward  hack
  -> 
```

，decoding ，prompt 。。

## 

""——benchmark、、。 RLHF ，base → SFT → RM → PPO ，。：，。

|       |                                             |                         |
| --------- | --------------------------------------------------- | ----------------------------- |
| Base  |  PPL、MMLU/CEval                              | ，      |
| SFT   | 、、val loss                    | ，PPO   |
|   | 、reward margin、-        | RM （、） |
| PPO   | KL 、reward 、、vs SFT  |  +      |

### Base ：

Base ，。。**（PPL）** ， held-out ：

$$
\text{PPL} = \exp\!\left(-\frac{1}{N}\sum_{t=1}^{N}\log p_\theta(y_t\mid y_{<t})\right)
$$

。** benchmark**（MMLU、CEval、HellaSwag）。" base"——SFT、PPO ，。

### SFT ： loss

SFT ，""。****： prompt（" JSON"、""），。**val loss**：， train loss  val loss 。SFT  PPO ， PPO 。

### ：

RM  PPO ， PPO 。****： RM  chosen  rejected ， 65-75%（ ~80%， 100%）。**reward margin**：chosen  rejected ，，。**-**：RM  Pearson ， > 0.5 。

### PPO ：

PPO 。**KL **， > 10  reference ， reward hacking。**reward **，。**** reward ， RM 。： benchmark  base、 SFT、。

：base  PPL  SFT ，SFT  RM 。，。

##  Benchmark

RLHF ：****。 HELM、MMLU  MT-Bench，：

|      |                           |          |
| -------- | --------------------------------- | ---------------- |
|  |  JSON / Markdown /  |  |
|  | 、、        |  |
|  | 、            |      |
|  | 、            |      |
|  | 、、      |    |

 smoke test。，。

```python
# ==========================================
# ： SFT  RLHF
# ==========================================
from dataclasses import dataclass

@dataclass
class EvalItem:
    prompt: str
    category: str
    checker: callable


def run_regression_eval(model, tokenizer, eval_items):
    results = []
    for item in eval_items:
        output = generate_answer(model, tokenizer, item.prompt)
        passed, reason = item.checker(output)
        results.append({
            "category": item.category,
            "passed": passed,
            "reason": reason,
            "output": output,
        })
    return results


def summarize_by_category(results):
    summary = {}
    for row in results:
        bucket = summary.setdefault(row["category"], {"ok": 0, "total": 0})
        bucket["total"] += 1
        bucket["ok"] += int(row["passed"])

    return {
        category: bucket["ok"] / bucket["total"]
        for category, bucket in summary.items()
    }
```

、 decoding ，。“”，。

 50  200 。，：

```json
{
  "id": "format-json-001",
  "category": "format_following",
  "prompt": " JSON， name  reason。",
  "checker": "valid_json_with_keys",
  "risk": ""
}
```

：

|            |                            |  |
| -------------- | ------------------------------ | ---------- |
|      | ，       |    |
| badcase  | ， |    |

，badcase 。。

## 

RLHF ， pairwise comparison。 prompt， SFT  RLHF ， judge 。

```python
# ==========================================
# Pairwise 
# ==========================================
judge_prompt = """
。。

：
1. 
2. 、
3. 
4. 

：
{prompt}

 A：
{answer_a}

 B：
{answer_b}

 JSON：
{{"winner": "A"  "B"  "tie", "reason": ""}}
"""
```

，A/B 。 judge ， judge ，。， human eval： 100  prompt， prompt  2-3 ，。

：

|         | Win | Lose | Tie | Win Rate |
| ----------- | --- | ---- | --- | -------- |
| RLHF vs SFT | 58  | 27   | 15  | 68.2%    |
| SFT vs Base | 72  | 14   | 14  | 83.7%    |

 win rate  prompt、 judge、 decoding 。。

### 

 20  prompt， 12 ， 60% ，。，。，：

1. ，。
2. Tie ，。
3. 。

 bootstrap ：

```python
def bootstrap_win_rate_ci(outcomes, n_boot=2000, seed=0):
    """
    outcomes: ["win", "lose", "tie", ...]
    tie  0.5 。
    """
    import random
    random.seed(seed)

    scores = [1.0 if x == "win" else 0.5 if x == "tie" else 0.0 for x in outcomes]
    rates = []
    for _ in range(n_boot):
        sample = [random.choice(scores) for _ in scores]
        rates.append(sum(sample) / len(sample))

    rates.sort()
    return {
        "win_rate": sum(scores) / len(scores),
        "ci_low": rates[int(0.025 * n_boot)],
        "ci_high": rates[int(0.975 * n_boot)],
    }
```

 95% ， 45%  72%，“RLHF  SFT”。：，。

### LLM-as-Judge 

LLM judge ，。：

|          |                      |               |
| ------------ | ------------------------ | --------------------- |
|      |  A  B            |           |
|      |              | rubric    |
|      |  Markdown      |       |
|      |  |  judge    |
|  |    |  rubric |

：

```json
{
  "prompt_id": "pref-042",
  "answer_a_model": "rlhf",
  "answer_b_model": "sft",
  "order_seed": 17,
  "judge_winner": "A",
  "judge_reason": "A  KL ，。",
  "human_checked": false
}
```

， judge 。

## 

 judge “”，。，：

- Reward 。
- Reward  SFT 。
- 。
- 。
- Judge  tie 。
- 、、、。

，“”。

|              |                                                    |
| ---------------- | ------------------------------------------------------ |
| prompt           |                                                |
| sft_answer       | SFT                                            |
| rlhf_answer      | RLHF                                           |
| rm_score_delta   | RLHF                                           |
| human_preference |                                        |
| issue_tags       | length_hack / repetition / hallucination / unsafe / ok |
| note             |                                                |

“RM ，”， RM ， PPO 。

 rubric，：

|    | 0            | 1                  | 2          |
| ------ | -------------- | -------------------- | ------------ |
|  |        |              |      |
|  |      |      |  |
|  |      |              |    |
|  |  |  |    |
|  |      |              |    |

（、、、），。，，。

## Reward hacking 

Reward hacking ： reward ，。“”，； RM  hack ，。

：

|                      |                               |           |
| ------------------------ | --------------------------------- | ------------- |
| reward     |           | length hack   |
|          |               | mode collapse |
| judge  RM  | RM ，/ judge  | RM  |

Reward hacking ， issue tag：

|      |                          |                   |
| -------- | ---------------------------- | ------------------------- |
|  | ， | length-reward       |
|  |              | n-gram / phrase frequency |
|  | 、 |     |
|  |    | fact-check /      |

```python
# ==========================================
# Reward hacking 
# ==========================================
def reward_hacking_signals(rows):
    """
    rows: [{"reward": float, "text": str}, ...]
    。
    """
    import numpy as np
    from collections import Counter

    rewards = np.array([r["reward"] for r in rows])
    lengths = np.array([len(r["text"]) for r in rows])
    length_corr = np.corrcoef(rewards, lengths)[0, 1]

    phrases = Counter()
    for row in rows:
        words = row["text"].split()
        phrases.update(" ".join(words[i:i + 4]) for i in range(max(0, len(words) - 3)))

    return {
        "length_reward_corr": float(length_corr),
        "top_phrases": phrases.most_common(5),
        "length_hack_warning": abs(length_corr) > 0.7,
    }
```

，：“”，“”。

：“”， reward、、， KL 。， [8.8 ](./extended-practice)。

### Reward hacking 

“reward ”，：

```text
1.  reward 
2.  reward 
3. 、、
4.  reward /
5.  judge 
6.  RM  rejected：、、
```

 reward hacking ，、 rubric、 RM、 KL 。

## 

 PPO 。 step  checkpoint，：

- `reward_mean`：RM 。
- `kl_mean`： reference  KL。
- `response_length`：。
- `distinct_ngram`：。
- `judge_win_rate`： pairwise 。
- `regression_score`：。

 reward ， reward 、KL 、、。 reward ， RM 。

 checkpoint ：

| step | reward | KL   | len | distinct-4 | reg score | judge win | note                |
| ---- | ------ | ---- | --- | ---------- | --------- | --------- | ------------------- |
| 0    | 0.12   | 0.00 | 156 | 0.83       | 0.78      | 50%       | SFT             |
| 200  | 0.18   | 0.04 | 162 | 0.82       | 0.78      | 54%       |                 |
| 400  | 0.31   | 0.09 | 210 | 0.74       | 0.76      | 56%       |           |
| 600  | 0.45   | 0.18 | 330 | 0.51       | 0.70      | 49%       |  reward hacking |

 reward 。 checkpoint 。

## 

：

|                  |                                    |
| -------------------- | -------------------------------------- |
| SFT vs Base  |  50%                           |
| RLHF vs SFT  |  55%，             |
|  benchmark       |  SFT  95%                      |
|          |  SFT  1.3 ， |
|                |                              |
|            |                      |

，。：，，。

## Reward Hacking 

 reward hacking，。，””。

### 

```python
def flawed_reward(prompt: str, response: str) -> float:
    “””
    。
    ：””””，。
    “””
    length_score = len(response) / 100.0

    format_score = 0.0
    if “- “ in response or “1.” in response:
        format_score += 0.5
    if “**” in response:
        format_score += 0.5

    politeness_score = 0.0
    for phrase in [“”, “”, “”, “”]:
        if phrase in response:
            politeness_score += 0.3

    return length_score + format_score + politeness_score
```

”、、”，：、、、。PPO ，、、。

。 prompt：” PPO  KL 。”

|                                                                                                                       |   |  |  |  |  |
| ------------------------------------------------------------------------------------------------------------------------- | ------- | ------ | -------- | ---- | -------- |
| “KL ， PPO 。”                                                                  |  0.32 | 0.0    | 0.0      | 0.32 |  |
| “。：- ****：PPO 。- ****：KL 。- ****：。” |  0.75 | 1.0    | 0.6      | 2.35 |  |

。 PPO ，。

### 

，：

|               |                | Reward hacking             |
| ----------------- | ---------------------- | ------------------------------ |
| `reward_mean`     |                | ， |
| `response_length` |  |  reward          |
| `distinct_ngram`  |            | ， |

：

| step | reward | length | distinct-4 | KL   |      |
| ---- | ------ | ------ | ---------- | ---- | ------------ |
| 0    | 0.8    | 120    | 0.82       | 0.00 | SFT  |
| 50   | 1.4    | 180    | 0.76       | 0.04 |    |
| 100  | 2.2    | 310    | 0.61       | 0.09 |    |
| 150  | 3.1    | 520    | 0.42       | 0.18 |    |
| 200  | 4.0    | 760    | 0.31       | 0.27 |  |

 reward，；，。

### 

””，：

```python
def safer_reward(prompt: str, response: str) -> float:
    helpfulness = judge_helpfulness(prompt, response)
    correctness = judge_correctness(prompt, response)
    format_score = validate_required_format(prompt, response)
    repetition_penalty = ngram_repetition_rate(response, n=4)
    length_penalty = max(0, len(response) - target_max_length(prompt)) / 400

    return (
        0.40 * helpfulness
        + 0.35 * correctness
        + 0.15 * format_score
        - 0.05 * repetition_penalty
        - 0.05 * length_penalty
    )
```

 helpfulness  correctness ，format ，length ，repetition 。 PPO-RLHF  KL ， SFT reference 。

## 

 RM ，。， RM ：

```json
{
  “prompt”: “ PPO  KL 。”,
  “chosen”: “KL ， PPO 。”,
  “rejected”: “。：PPO ，KL ，...”,
  “tags”: [“length_hack”, “template_hack”],
  “source”: “ppo_badcase”
}
```

：，，。

：

```text

  ->  badcase、、
  -> 
  ->  SFT / preference 
  -> 
  ->  SFT、RM  PPO-RLHF
  -> 
  -> 
```

””，。badcase ：

|             |                  |                      |
| --------------- | -------------------- | ---------------------------------- |
| `length_hack`   |  |  chosen、 rejected |
| `template_hack` |      |  chosen、 rejected     |
| `hallucination` |        | 、         |
| `over_refusal`  |      |                      |
| `under_refusal` |      |                    |

## 

RLHF ：

1. ？
2. ？
3.  reward ？

 reward ， reward hacking 。 + reward hacking  + ， RLHF 。

 RLHF ：base model  assistant，SFT ，RM ，PPO ，。， [8.8 ](./extended-practice)； RLHF ， RM、Critic ——[](../chapter09_alignment/intro)。

## 

1.  30  prompt ，、、、、 5 。
2.  10  pairwise  win rate， tie  0.5 。
3.  rubric， reward ””。
4.  `flawed_reward`，””， 3  stress case。
5. ： badcase ，？
