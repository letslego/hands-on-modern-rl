# 8.3 SFT：

## 

****

-  SFT （Behavior Cloning）。
-  SFT 、、loss mask、packing、。
-  Reward Model  DPO  pairwise 。

****

$$
\mathcal{L}_{SFT}(\theta)
= -\sum_{t=1}^{T} m_t \log \pi_\theta(y_t \mid x, y_{<t})
\quad \text{（SFT： assistant token  loss）}
$$

> **SFT  mask：**
>
> - $x$：、system prompt、。
> - $y_t$：assistant  $t$  token。
> - $m_t$：loss mask。assistant token  1， assistant token  0。

$$
\mathcal{D}_{pref}=\{(x,y_w,y_l)\}
\quad \text{（： prompt  chosen  rejected）}
$$

****

SFT “”，，：，， assistant  token；，SFT ，。 loss mask ， Reward Model 。

> RLHF ：**，**。  
> ：**SFT “”？**

 pretrained base model 。， assistant。 RLHF， PPO， artifact："" SFT ，。？——（BC）（IRL）。

 BC  IRL，RLHF ，。SFT  BC，RM  IRL ， RL 。：（BC → SFT），（IRL → RM），（RL → PPO）。

## 8.3.1 

。（RL），——，；，。""，。。

 RL ， $(s_t, a_t)$ -， $\pi_\theta(a|s)$ 。，——。[^bc]

```python
# ==========================================
# （BC）：
# ==========================================
import torch
import torch.nn as nn

class BCPolicy(nn.Module):
    """：，"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)

# BC ：
def train_bc(policy, expert_states, expert_actions, epochs=100):
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # ，
        pred = policy(expert_states)
        loss = criterion(pred, expert_actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
```

 4 ——BC 。 SFT ： token ""，SFT -，""。

### BC ：

BC ，。，""。，。————（）。，，。，，……**（Distribution Shift）**——。

```
:  s₀ → s₁ → s₂ → s₃ → ... ()

:  s₀ → s₁' → s₂' → s₃' → ... (，)
                ↑
            
            →  →  → 
```

 LLM ，。SFT ，，，，。 SFT ——""， RL。

， DAgger：，“”，。[^dagger]

## 8.3.2 

BC 。""，""。，（BC），——，？

（IRL）。 RL ， $R(s,a)$ ，。IRL ： $\mathcal{D} = \{\tau_1, \tau_2, \ldots\}$， $R(s,a)$ ， RL 。

```mermaid
flowchart LR
    subgraph RL[""]
        R1[" R(s,a)"] --> RL1["RL "]
        RL1 --> Pi1[" π*"]
    end

    subgraph IRL[""]
        D[" D"] --> IRL1["IRL "]
        IRL1 --> R2[" R̂(s,a)"]
    end

    style RL fill:#e3f2fd,stroke:#1565c0
    style IRL fill:#fff3e0,stroke:#e65100
```

IRL ：，，，……。[^irl]  LLM ，InstructGPT ——，，。[^instructgpt]  RLHF  RM ：IRL （），。

###  BC-IRL  SFT-RM-RL 

 BC  IRL  RLHF ，：

|               | LLM          |                           |                |                  |
| ----------------- | ------------------ | ----------------------------- | ------------------ | -------------------------- |
| BC（）    | SFT（）    | -                   |  | """" |
| IRL（） | RM（） |  (chosen, rejected) |    | """"   |
| RL（）    | PPO            | SFT  + RM             |    | """"       |

：BC ""，；IRL ""，；RL  IRL ， BC 。。

## 8.3.3 SFT 

，**70% ，**。——，、、。 SFT 。

### Self-Instruct：

Self-Instruct “，”。[^self_instruct]

Self-Instruct ，，。：

```python
# ==========================================
# Self-Instruct：
# ==========================================
seed_instructions = [
    {"instruction": "", "response": "..."},
    {"instruction": " Python ", "response": "def quicksort(arr): ..."},
    # ... 
]

# （ GPT-4）
prompt = """
：
{seed_examples}

。：
1. 
2. 、
3. （//）
"""

# ：、、
def filter_generated(data, similarity_threshold=0.85):
    """"""
    filtered = []
    for item in data:
        #  embedding 
        sim = max(cosine_similarity(item, existing) for existing in filtered)
        if sim < similarity_threshold:
            filtered.append(item)
    return filtered
```

### Evol-Instruct：

Evol-Instruct（ WizardLM “”）“”“”。[^wizardlm]

Evol-Instruct  Self-Instruct ""——，""""""：

|              |                        |                                                           |
| -------------------- | -------------------------- | ------------------------------------------------------------- |
| （Deepen）       |                | "" → ""             |
| （Widen）        |                | " Python " → " Python 、、" |
| （Concretize） |                | "" → " O(log n) " |
| （Simplify）     | ， |                                                       |

### 

，。：

```
 1（ 20% ）：90%  + 10%     → 
 2（20%-60% ）：30%  + 50%  + 20%   → 
 3（ 40% ）：10%  + 40%  + 50%   → 
```

。（、、、、）。（），。

### ：SFT “”

“”，，，**、mask 、**。 SFT ，。

#### 1) ：， token

 SFT  `(instruction, response)`，：

```json
{
  "messages": [
    { "role": "system", "content": "。" },
    { "role": "user", "content": " PPO 。" },
    { "role": "assistant", "content": "PPO ..." }
  ],
  "meta": { "source": "human", "task": "explain", "lang": "zh" }
}
```

：

- ****：system/user/assistant 、、（stop tokens），“”。
- ****：`source/task/lang/difficulty` 、 badcase 。
- ****： schema ，（ NaN ）。

#### 2)  loss mask：“ token”

： user/system  token  loss 。** assistant  token **（ token `labels=-100`）。，“”。[^sft_mask]

，：、 loss，“”，“”。

。 tokenizer ，：

```text
<system> 
<user>  PPO
<assistant> PPO 
```

 8  token：

|  | token          |            | label             |
| ---- | -------------- | -------------- | ----------------- |
| 1    | `<system>`     | system         | -100              |
| 2    | ``     | system         | -100              |
| 3    | `<user>`       | user           | -100              |
| 4    | ` PPO`   | user           | -100              |
| 5    | `<assistant>`  | assistant  | -100  |
| 6    | `PPO`          | assistant      | token id          |
| 7    | ``       | assistant      | token id          |
| 8    | `` | assistant      | token id          |

`-100`  PyTorch  ignore index。：， loss。，。

 mask ：

```python
def build_labels(input_ids, assistant_mask, ignore_index=-100):
    """
    assistant_mask:  input_ids ，assistant token  1， 0。
    """
    labels = input_ids.clone()
    labels[assistant_mask == 0] = ignore_index
    return labels
```

，“”，“”。

#### 3) Packing ：、、

，。：

- ** packing**：， padding。；。
- **packing**：（），。； mask，“”。

：“ packing +  mask”、， packing 。“”。

### SFT 

，SFT ：

```text

  -> schema 
  ->  chat template  loss mask
  ->  SFT model
  ->  prompt 
```

：

|          |                                 |                        |
| ------------ | ------------------------------------- | ------------------------------ |
| `train_loss` |                 | ， mask  |
| `eval_loss`  |                       | train  eval              |
|    |  JSON、Markdown、 |              |
|        |                     | n-gram               |
|      |                       | loss               |

： SFT loss。SFT loss ，。，loss ，“、”。

## 8.3.4 

 DPO  RLHF （RM ）。 $(x, y_w, y_l)$： prompt $x$，，（ AI） $y_w$（chosen） $y_l$（rejected）。

### ：Bradley-Terry  pairwise loss

：`chosen` vs `rejected`？： prompt，“” $r_\phi(x, y)$ ， logistic （Bradley-Terry）。[^bt]

RM  pairwise loss ：

$$
\mathcal{L}_{RM} = - \mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))\right]
$$

： RM “”， prompt 。

### 

|          |              |  |              |        |
| ------------ | ---------------- | ---- | ---------------- | -------------- |
|      | （$0.5-5$/） |    | （） |  |
| AI   |                |    |                |  |
| LLM-as-Judge |                |  |                |  |
|      |              |    |              |    |

****。。—— prompt 。 2-3 ，。

**AI **。（GPT-4、Claude、Gemini ），。 GPT-4  A  1，Claude  3，。

**LLM-as-Judge** （ GPT-4），。（prompt）—— Judge （、、）。

```python
# ==========================================
# LLM-as-Judge：
# ==========================================
judge_prompt = """
。：
1. ：
2. ：
3. ：
4. ：

: {prompt}
 A: {response_a}
 B: {response_b}

 JSON ：
{{"winner": "A"  "B", "reason": ""}}
"""

# 
def construct_preference_pairs(prompt, responses, judge_outputs):
    """ Judge """
    pairs = []
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            if judge_outputs[i]['score'] > judge_outputs[j]['score']:
                pairs.append((prompt, responses[i], responses[j]))  # (x, y_w, y_l)
            else:
                pairs.append((prompt, responses[j], responses[i]))
    return pairs
```

****——/、（edit-and-resend）、。、，。

## 8.3.5 

，。：

```mermaid
flowchart TD
    Raw[""] --> D1["（）"]
    D1 --> D2["（）"]
    D2 --> D3["（）"]
    D3 --> D4["（）"]
    D4 --> D5["（）"]
    D5 --> Clean[""]

    style Raw fill:#ffebee,stroke:#c62828
    style Clean fill:#e8f5e9,stroke:#2e7d32
```

**** embedding —— embedding （ 0.85），。****：（ 50 ）、（n-gram  60%）、。****、、。****——，。

“”：**（decontamination）**。 benchmark ，。，。[^decontam]

<details>
<summary>： Self-Instruct ，？</summary>

""——，，。，""——。 Self-Instruct ，。

****。 Markdown ， Markdown ，。 RLHF 。

</details>

## 8.3.6 RLHF 

 SFT ， RLHF ：

```mermaid
flowchart LR
    subgraph S1["：SFT"]
        SFTData[""] --> SFTTrain[""]
        SFTTrain --> SFTModel["SFT "]
    end

    subgraph S2["：RM"]
        PrefData[""] --> RMTrain[""]
        RMTrain --> RM[""]
    end

    subgraph S3["：RL"]
        SFTModel --> PPO["PPO "]
        RM -->|""| PPO
        PPO --> Final[""]
    end

    SFTData -.->|""| PrefData

    style S1 fill:#e3f2fd,stroke:#1565c0
    style S2 fill:#fff3e0,stroke:#e65100
    style S3 fill:#e8f5e9,stroke:#2e7d32
```

**（SFT）** -，。SFT ：$\mathcal{L}_{SFT} = -\mathbb{E}_{(x,y)}[\log \pi_\theta(y|x)]$。 LLM 。

**（RM）**  $(x, y_w, y_l)$，。 IRL ——，。

**（RL）**  SFT  RM，。PPO  RM （ 7 ）。 SFT ，。

——。SFT ，，RM ；，RM ，RL 。 RLHF 。

， RLHF ""——。，。——[Reward Model：](./reward-function-design)。

## 

SFT  RLHF ，“”。，SFT  LLM ：，。，SFT 、loss mask、、、。

。， judge 。， Reward Model、DPO 。

## 

1.  system、user、assistant  SFT ， token  loss。
2.  prompt  chosen  rejected， rejected ，“”。
3. ，“”。

## 

[^bc]: Pomerleau D A. [ALVINN: An Autonomous Land Vehicle in a Neural Network](https://papers.nips.cc/paper_files/paper/1988/hash/812b4ba287f5ee0bc9d43bbf5bbe87fb-Abstract.html), NeurIPS 1988.（）

[^dagger]: Ross S, Gordon G, Bagnell D. [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html), AISTATS 2011.（DAgger：）

[^irl]: Ng A Y, Russell S. [Algorithms for Inverse Reinforcement Learning](https://ai.stanford.edu/~ang/papers/icml00-irl.pdf), ICML 2000.（IRL ）

[^instructgpt]: Ouyang L, Wu J, Jiang X, et al. [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155), 2022.（RLHF ：SFT → RM → RL）

[^self_instruct]: Wang Y, Kordi Y, Mishra S, et al. [Self-Instruct: Aligning Language Models with Self Generated Instructions](https://arxiv.org/abs/2212.10560), 2022.

[^wizardlm]: Xu C, Sun Q, Zheng K, et al. [WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/abs/2304.12244), 2023.

[^bt]: Bradley R A, Terry M E. [Rank Analysis of Incomplete Block Designs](https://www.jstor.org/stable/2334029), Biometrika 1952.（Bradley-Terry ）

[^sft_mask]: Hugging Face TRL Documentation. [SFTTrainer](https://huggingface.co/docs/trl/sft_trainer).（SFT  assistant token  loss ）

[^decontam]: OpenAI. [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165), 2020.（；）
