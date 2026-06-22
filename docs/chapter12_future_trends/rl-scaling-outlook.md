# 12.6 RL Scaling ——RL ？

 RL  PPO  GRPO  DAPO/RLVR 。 Critic， RM，。：**RL ？？**

2025 ：**RL **。 RL ，。 RL ：（Online vs Offline）、RL Scaling 、 Test-time Scaling 。

## 

 DPO（Offline） GRPO/DAPO（Online），：**？**

|                | Offline (DPO)        | Online (PPO/GRPO)        | Semi-Online           |
| -------------- | -------------------- | ------------------------ | --------------------- |
| ****   |  |          |  +  |
| ****   | （）   | （） |                   |
| ****   |        |                |                   |
| **** | （）   | （）       |                   |
| ****   |                    |                        |                   |
| ****   | DPO, KTO, SimPO, IPO | PPO, GRPO, DAPO          | Iterative DPO, RLOO   |
| ****       |          |                |  +    |

### 

：

1. **：DPO **。 DPO 。DPO 、， DPO ，， PPO/GRPO 。
2. **：GRPO **。 DPO ， GRPO 。GRPO  DPO 。
3. **：DAPO **。， DAPO  Token 。

```python
# ==========================================
# （）
# ==========================================

# ---- Offline (DPO) ----
# ：，
# dpo_trainer = DPOTrainer(model, ref_model, dataset=preference_pairs)
# dpo_trainer.train()

# ---- Online (GRPO) ----
# ：， Critic
# grpo_trainer = GRPOTrainer(model, reward_fn=rule_based_reward, k=8)
# grpo_trainer.train()

# ---- Semi-Online (Iterative DPO) ----
# ：， DPO 
# for iteration in range(num_iterations):
#     new_data = model.generate_and_label(prompts)  #  + 
#     dpo_trainer.train_on(new_data)                 # DPO 
#     model = dpo_trainer.get_updated_model()        # 

print("：")
print("  ？ → DPO ")
print("  DPO ？ → GRPO ")
print("  ？ → Iterative DPO（）")
print("  ？ → DAPO（ + Token）")
```

## RLMT：""

（DPO/GRPO/DAPO） RLVR ：****。——""，、？

2025  "Language Models that Think, Chat Better" ， **RLMT（Reinforcement Learning with Model-rewarded Thinking）**。

### 

|  |  |              |      |                  |
| ---- | ------ | -------------------- | ------------ | -------------------- |
| RLHF |      |      |      | ，     |
| RLVR |      | /        | /    |    |
| RLMT | **** | **** | **** |  |

RLHF ，；RLVR ，（）。RLMT ：** RLVR ""， RLHF **——。

### RLMT 

RLMT ， DeepSeek-R1  SFT  Zero ：

**：SFT  + RLMT**

1.  Gemini/GPT-4 " + "，""
2.  GRPO ，

**：RLMT-Zero（）**

 SFT， RLMT 。：

-  **7K  prompts**
- Llama-3.1-8B  + RLMT-Zero
- **** 2500  Llama-3.1-8B-Instruct

"" SFT ——RL ， DeepSeek-R1 。

### ： > 

 Llama-3.1-8B  Qwen-2.5-7B ：

- ****（AlpacaEval2 / WildBench / ArenaHardV2） 3–7 
- **、、** 1–3 
- Llama-3.1-8B-Instruct + RLMT ** GPT-4o**， Claude 3.7 Sonnet
-  10  Llama-3.1-70B、Qwen2.5-72B

：**RL ""**。

### ""

 RLMT ：

|       |                             |                      |
| --------- | ----------------------------------- | ------------------------ |
| SFT   | 、、          |        |
| RLMT  | →→→ |  |

，——， RL ： →  → 。

### RLMT 

```python
# ==========================================
# RLMT vs RLVR （）
# ==========================================

# ---- RLVR（）----
#  = （）
# def rlvr_reward(response, question):
#     answer = extract_answer(response)
#     return 1.0 if answer == ground_truth else 0.0

# ---- RLMT（）----
#  = （）
# def rlmt_reward(response, question):
#     # response  <think</think + 
#     return preference_reward_model(question, response)

# ：
# 1. RLMT  response  = <think</think + 
# 2.  RM，
# 3.  prompts ，

print("RLMT ：")
print("  —— RM ")
print("   prompts ")
print("  GRPO  PPO、DPO，")
print("   RLMT ，")
```

### RLMT 

RLMT  9  RLVR  7  RLHF ：

|                 |  RLMT              |
| ----------------------- | ---------------------------- |
| RLVR （Ch8）  | "" |
| RLHF （Ch7）  |  RM        |
| GRPO （Ch8）  | RLMT     |
| DeepSeek-R1-Zero（Ch8） | RLMT-Zero      |

RLMT ：**""，**。 RL ——，""。

<details>
<summary>： RLVR ， RLMT ？</summary>

****。RLVR ""——""。，""，。

RLMT 。 RM ""（、、），""。，——""，""。

： RM ，。

</details>

## RL Scaling：

2025 ：**RL **。DeepSeek-R1 ，，RL  scaling  SFT 。， pass@1 ，。

### RL Scaling 

|          |                                |                   |                        |
| ------------ | ---------------------------------- | ------------------------- | ------------------------------ |
| **** |  prompt          |  +  |              |
| **** |  prompt （k ） | k  4  16  64  | ， |
| **** |  RL                  |  KL     | Pass@1 ，      |

```mermaid
flowchart LR
    subgraph scaling ["RL Scaling "]
        D[" ↑\n\n"]
        S[" ↑\nk: 4 → 16 → 64\n"]
        T[" ↑\n\n"]
    end

    D --> Result["\n\n"]
    S --> Result
    T --> Result

    subgraph caveat [""]
        C1["Prompt \n"]
        C2["KL \n"]
        C3["\n"]
    end

    style D fill:#e3f2fd,stroke:#1976d2
    style S fill:#fff3e0,stroke:#f57c00
    style T fill:#e8f5e9,stroke:#2e7d32
    style Result fill:#fce4ec,stroke:#c62828
```

： prompt 。，——，。DeepSeek-R1 ，。

### Agentic RL  Scaling Law

 RL  scaling。 RL  Agentic （ 9 ），scaling 。ZeroTIR ****，：、****。—— 100 ，；，。****：，""。ZeroTIR  NeurIPS 2025 ， 9  Code Agent 。

## Test-time Scaling：

 RL Scaling（）：**""**。

"Prompt → "。Test-time Scaling "Prompt →  → // → "。

|                        |                              |          |                 |
| -------------------------- | -------------------------------- | ---------------- | ----------------------- |
| **Best-of-N **         |  N ， reward   |  N     | ，          |
| ****               |  N ，  |  N     | /（） |
| **MCTS / Tree of Thought** | ， | （） |             |
| **Verifier-guided**        |      |              | /               |

### RL  Test-time Scaling 

：**RLVR  test-time ，？**

- ：RL ， RL ，。
- ： RL ，（N ），——RL ""。
- ： RL ——。 RL ""，。

## PRM vs ORM：

，：**（），（）？**  PRM（Process Reward Model） ORM（Outcome Reward Model）。

### ORM（Outcome Reward Model）

ORM ：，。——。——7 ，。

### PRM（Process Reward Model）

PRM ：？？...。，。——。

### PRM 

|      | GSM8K  | MATH  |  |
| -------- | ------------ | ----------- | -------- |
|  ORM   | ~82%         | ~40%        |        |
|  PRM   | ~85%         | ~45%        |      |
| ORM + RL | ~88%         | ~50%        |        |
| PRM + RL | ~90%         | ~55%        |      |

PRM （ MATH  ORM  5 ），。OpenAI  PRM800K ，。

###  PRM 

，：

```python
# ==========================================
#  PRM：
# ==========================================
def auto_prm(model, prompt, reasoning_steps, num_samples=32):
    """
    

    ： i ， N 
     →  i ""
    """
    step_scores = []

    for i in range(len(reasoning_steps)):
        #  i ，
        correct_count = 0
        for _ in range(num_samples):
            #  i 
            new_completion = model.generate(
                prompt + reasoning_steps[:i+1],
                temperature=0.7  # ，
            )
            # 
            if check_answer_correct(new_completion):
                correct_count += 1

        #  i  = 
        step_scores.append(correct_count / num_samples)

    return step_scores

# 
# reasoning_steps = [" x = ", "x = 15 - 3 - 5", "x = 7"]
# step_scores = [0.85, 0.90, 1.00]
#  85% ，
```

 PRM ：**，""**。 $i$  $N$ ，—— $i$ ""。（ $N$ ），，。

## 

 8  RL ：

```mermaid
flowchart TD
    subgraph strategy [""]
        S1["PPO\nActor + Critic + Ref + RM\n4 "]
        S2["GRPO\nActor + Ref\n2 （ Critic）"]
        S3["DAPO\n + Token\n GRPO"]
    end

    subgraph reward [""]
        R1["RLHF\n + RM \n"]
        R2["RLVR\n\n"]
    end

    subgraph paradigm [""]
        P1["Offline (DPO)\n"]
        P2["Online (GRPO)\n"]
        P3["Semi-Online\n"]
    end

    S1 --> S2 --> S3
    R1 --> R2

    style S1 fill:#fce4ec,stroke:#c62828
    style S2 fill:#fff3e0,stroke:#f57c00
    style S3 fill:#e8f5e9,stroke:#2e7d32
    style R1 fill:#fce4ec,stroke:#c62828
    style R2 fill:#e8f5e9,stroke:#2e7d32
```

、：

- ****：PPO → GRPO → DAPO（）
- ****：RLHF → RLVR（）
- ****：Offline → Online → Semi-Online（）

RL ：**RL Scaling**（） **Test-time Scaling**（）。，。 PRM ，， RL 。

<details>
<summary>：RL Scaling  Test-time Scaling ？</summary>

：

- ****（）， RL Scaling——，，。，；，。

- ****（、，）， Test-time Scaling—— Best-of-N  MCTS ""，。

- ****（、），。RL Scaling ""——。

，。（ DeepSeek-R1） RL Scaling（ GRPO ） Test-time Scaling（ Best-of-N ），。

</details>

，RL Scaling 。——****： Teacher  log-prob ， 1/10  RL 。——[](../chapter09_grpo_rlvr/on-policy-distillation)。
