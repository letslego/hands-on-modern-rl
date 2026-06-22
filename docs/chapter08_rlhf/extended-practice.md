# 8.8 ：Reward Hacking 

## 

****

- ， reward hacking 。
-  reward、、、KL ，。
-  badcase 、、、。

****

$$
R_{bad}(x,y)=0.01|y|+0.5\cdot\mathbb{1}[\text{has\_list}(y)]
+0.3\cdot\mathbb{1}[\text{has\_polite\_phrase}(y)]
\quad \text{（：“”“、”）}
$$

$$
R_{safe}(x,y)=
0.4R_{helpful}+0.3R_{correct}+0.2R_{format}
-0.05R_{length}-0.05R_{repeat}
\quad \text{（：）}
$$

> ****
>
> Reward hacking “”，。，、、。

8.1-8.7  RLHF ：SFT ，RM ，PPO ，。： reward hacking ，。

## 

 3 ：。 RLHF 。“” RM、judge、。

：

>  reward ，。

Reward hacking 。，“”。：

1.  reward 。
2. 。
3.  RLHF 。

## 

Reward hacking ，。：、，“”。

```python
def flawed_reward(prompt: str, response: str) -> float:
    """
    。
    ：“”“”，。
    """
    length_score = len(response) / 100.0

    format_score = 0.0
    if "- " in response or "1." in response:
        format_score += 0.5
    if "**" in response:
        format_score += 0.5

    politeness_score = 0.0
    for phrase in ["", "", "", ""]:
        if phrase in response:
            politeness_score += 0.3

    return length_score + format_score + politeness_score
```

“、、”，：

```text




```

PPO  GRPO ，、、。，。

### 

 prompt：

```text
 PPO  KL 。
```

 A：

```text
KL ， PPO 。
```

 B：

```text
。：
- ****：PPO 。
- ****：KL 。
- ****：。
```

 `flawed_reward` ：

|  |   |  |  |  |  |
| ---- | ------- | ------ | -------- | ---- | -------- |
| A    |  0.32 | 0.0    | 0.0      | 0.32 |  |
| B    |  0.75 | 1.0    | 0.6      | 2.35 |  |

 B。 PPO ， B 。

## 

 RLHF。、 prompt、。

|       |                                      |
| --------- | ---------------------------------------- |
|       |  SFT ， chat model |
| prompt  | 50  200                              |
|   |  `max_new_tokens=256`              |
|       | `flawed_reward`                          |
|       |  prompt  SFT             |
|       | reward、、、           |

，。

：

```python
for step in range(num_steps):
    prompts = sample_prompts(prompt_pool)
    responses = actor.generate(prompts, max_new_tokens=256)

    rewards = [flawed_reward(p, r) for p, r in zip(prompts, responses)]
    kl = compute_kl(actor, reference, prompts, responses)
    total_rewards = [r - beta * k for r, k in zip(rewards, kl)]

    ppo_update(actor, critic, prompts, responses, total_rewards)

    if step % eval_interval == 0:
        log_reward_hacking_metrics(step, prompts, responses, rewards, kl)
```

 PPO， decoding ， `flawed_reward` 。。

## 

， reward。：

|               |                | Reward hacking             |
| ----------------- | ---------------------- | ------------------------------ |
| `reward_mean`     |                | ， |
| `response_length` |  |  reward          |
| `distinct_ngram`  |            | ， |

 PPO-RLHF ：

|              |                           |
| ---------------- | --------------------------------- |
| `kl_mean`        |  Actor  reference   |
| `judge_win_rate` |  judge  |

：

```python
def reward_hacking_report(rows):
    """
    rows: [{"reward": float, "text": str}, ...]
    """
    import numpy as np
    from collections import Counter

    rewards = np.array([row["reward"] for row in rows])
    lengths = np.array([len(row["text"]) for row in rows])
    length_corr = float(np.corrcoef(rewards, lengths)[0, 1])

    phrases = Counter()
    for row in rows:
        words = row["text"].split()
        phrases.update(" ".join(words[i:i + 4]) for i in range(max(0, len(words) - 3)))

    unique_4grams = len(phrases)
    total_4grams = sum(phrases.values())
    distinct_4 = unique_4grams / max(total_4grams, 1)

    return {
        "length_reward_corr": length_corr,
        "distinct_4": distinct_4,
        "top_phrases": phrases.most_common(5),
        "warning": length_corr > 0.7 or distinct_4 < 0.5,
    }
```

 hack，。， reward 、、。

## 

：

| step | reward | length | distinct-4 | KL   |      |
| ---- | ------ | ------ | ---------- | ---- | ------------ |
| 0    | 0.8    | 120    | 0.82       | 0.00 | SFT  |
| 50   | 1.4    | 180    | 0.76       | 0.04 |    |
| 100  | 2.2    | 310    | 0.61       | 0.09 |    |
| 150  | 3.1    | 520    | 0.42       | 0.18 |    |
| 200  | 4.0    | 760    | 0.31       | 0.27 |  |

 reward，；，。

：

```text
。：

1. ****，PPO 。
2. ****，KL 。
3. ****， KL  PPO 。
4. ****， PPO。

，，。
```

、、、，。 KL 。

## 

“”，：

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

：

|                             |                                      |
| ------------------------------- | ---------------------------------------- |
| helpfulness  correctness  | “”“” |
| format              |                    |
| length ，         |                            |
| repetition              |                              |
| target length  prompt       |            |

 PPO-RLHF  KL ， SFT reference 。， reward，、 benchmark、。

##  Rejected 

 RM ，。， RM ：

```json
{
  "prompt": " PPO  KL 。",
  "chosen": "KL ， PPO 。",
  "rejected": "。：PPO ，KL ，...",
  "tags": ["length_hack", "template_hack"],
  "source": "ppo_badcase"
}
```

：，，。

## 

RLHF 。：

```text

  ->  badcase、、
  -> 
  ->  SFT / preference 
  -> 
  ->  SFT、RM  PPO-RLHF
  -> 
  -> 
```

“”，。：

```python
def active_learning_cycle(model, eval_set, data_producer):
    errors = evaluate_and_collect_errors(model, eval_set)
    clusters = cluster_errors_by_type(errors)

    new_data = []
    for cluster in clusters.top_k(k=3):
        new_data.extend(data_producer.generate(
            task_type=cluster.type,
            difficulty=cluster.difficulty,
            num_samples=1000,
        ))

    cleaned = quality_gate(new_data)
    updated_model = train_on_new_data(model, cleaned)
    report = regression_eval(updated_model, eval_set)
    return updated_model, report
```

 `quality_gate` 、、、chosen/rejected 、。，， judge 。

## 

badcase ，。：

|             |                    |                      |
| --------------- | ---------------------- | ---------------------------------- |
| `length_hack`   |    |  chosen、 rejected |
| `template_hack` |        |  chosen、 rejected     |
| `hallucination` |          | 、         |
| `over_refusal`  |        |                      |
| `under_refusal` |        |                    |
| `format_fail`   | JSON// |  SFT               |
| `reasoning_gap` |        |                |

，。

## 

**Reasoning 。** ，。 prompt ，，，。，。

```text
 N 
  -> /
  ->  chosen
  -> 、、 rejected
  ->  RM / DPO / RLVR
  ->  GSM8K、MATH、HumanEval 
```

**Agent 。** Agentic RL ，。 chosen；、、，。，“”。

```text
Agent 
  -> /
  ->  chosen
  -> 
  -> 
  ->  SWE-bench、WebArena 
```

## 

 reward hacking ，：

|             |                      |
| --------------- | ------------------------ |
| reward          |        |
|             |          |
| distinct n-gram |  SFT       |
|         |      |
|           |          |
| stress case     | 、 |

 reward ，，。 reward 。RLHF ，“”，。

## 

Reward hacking ，。RLHF ，、。

：base model  assistant，SFT ，RM ，PPO ，。 RLHF ， RM、Critic 。

## 

1.  `flawed_reward`，“”， 3  stress case。
2.  reward hacking ： length、template、format、semantic ？
3. ： badcase ，？
