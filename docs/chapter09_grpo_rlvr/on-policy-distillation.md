# 9.6 （OPD）—— Teacher 

 RLVR  RM，、。：**， token **。（On-Policy Distillation, OPD）。

：student  prompt （""）；teacher ， student ， token ； token  student，。：student  RL **（policy）**， token **（action）**，teacher  token  log-prob ****。

OPD 。****：SFT / SeqKD  teacher ，GRPO  student ，OPD  student 。****：GRPO / RLVR —— 1 、 0 ， 2000 token  0  1；OPD  token ， teacher  token  log-prob。

|         |  |          |             |                   |
| ----------- | -------------- | -------------------- | ------------------- | ------------------------- |
| SFT / SeqKD |  teacher |  token       | token-level         | student     |
| PPO / GRPO  | student        | RM       |  sequence-level | ，        |
| DPO         |    | chosen / rejected  | sequence-pair-level |               |
| **OPD**     | **student**    | **teacher log-prob** | **token-level**     |  teacher  |

：**OPD  RL  on-policy 。**

## OPD 

### Teacher  OPD 

 teacher， teacher 。

|                     | teacher                                      |  | student          |                                                     |
| ------------------- | -------------------------------------------------- | -------------- | ------------------------ | ------------------------------------------------------- |
| **SFT / SeqKD**     | ：                             | teacher        |                |  teacher ，                 |
| **Off-policy ** | ：                       | teacher        |                |  teacher ，student  |
| **OPD**             | ： student ， token  | student        |  |  teacher                            |

，teacher ********，student  teacher 。OPD ：student ，， teacher 。

### OPD 

OPD ：

1.  prompt $x$，student $\pi_\theta$  $y \sim \pi_\theta(\cdot \mid x)$。
2.  student  $c_t=(x,y_{<t})$，teacher $q$  token $y_t$ 。
3.  teacher  student  token  student。

Google DeepMind  GKD ： teacher ， student  teacher ， forward KL、reverse KL、JSD 。[^gkd]

 reverse KL ，：

$$
\mathcal{L}_{\text{OPD}}(\theta)
= \mathbb{E}_{y \sim \pi_\theta}
\left[
\sum_t
\log \frac{\pi_\theta(y_t \mid c_t)}{q(y_t \mid c_t)}
\right]
$$

 student ：

$$D_{\text{KL}}(\pi_\theta(\cdot \mid c_t) \| q(\cdot \mid c_t))$$

， token ：

$$
r_t
= \log q(y_t \mid c_t) - \log \pi_\theta(y_t \mid c_t)
$$

teacher  student  token，$r_t$ ；teacher  token ，$r_t$ 。Thinking Machines Lab ， RL  KL regularizer model  teacher： student rollout， student log-prob， teacher  log-prob， reverse KL  per-token advantage。[^tml_opd]

 OPD  RL ：

| RL     | OPD                                            |
| ---------- | -------------------------------------------------------- |
|  $s_t$ | prompt +  $c_t$                                |
|  $a_t$ |  token $y_t$                                       |
|        | student $\pi_\theta$                                     |
|        | teacher  student token                     |
|    | student                                        |
|    |  $\log q(y_t \mid c_t)-\log \pi_\theta(y_t\mid c_t)$ |

：teacher ，。OPD " teacher "， teacher  student  student。， RL。

## ：distillation, off-policy, on-policy

 **distillation**、**off-policy / offline**  **on-policy** 。，。

**（distillation）**  teacher-student ： $q$  $\pi_\theta$。 teacher ，student ：

$$
\mathcal{L}_{\text{hard KD}}
= -\sum_t \log \pi_\theta(y_t^T \mid x, y_{<t}^T)
$$

 student  teacher  token $y_t^T$。 teacher ，""： student  token ， token 。

$$
\mathcal{L}_{\text{soft KD}}
= \sum_t D_{\text{KL}}\left(q(\cdot \mid c_t) \| \pi_\theta(\cdot \mid c_t)\right)
$$

。 teacher  "therefore" ，"so" ，"banana" 。 teacher ， teacher 。

**policy（）**  LLM "， token "：

$$
\pi_\theta(a_t \mid s_t)
\quad \Longleftrightarrow \quad
\pi_\theta(y_t \mid x, y_{<t})
$$

 $s_t$  prompt ， $a_t$  token。，（behavior policy）。。

**Off-policy** ： student ， $\mu$。 $\mu$  teacher、 checkpoint、、，。 teacher  off-policy：

$$
y \sim q(\cdot \mid x),
\quad
\text{update } \pi_\theta
$$

 teacher $q$， student $\pi_\theta$。、、； student 。

**Offline**  off-policy ：，，。SFT、DPO、 offline。：

|        |             |  |                         |
| ---------- | ----------------------- | ------------------ | --------------------------- |
| offline    |           |                | SFT、DPO、 SeqKD        |
| off-policy |  student  | ， | teacher 、 replay |
| on-policy  |  student    |                | PPO、GRPO、OPD rollout      |

，**offline  LLM  off-policy， off-policy  offline**。 teacher ， student， offline， off-policy， teacher， student。

**On-policy** ： student  student。：

$$
y \sim \pi_\theta(\cdot \mid x),
\quad
\text{update } \pi_\theta
$$

。student ，，。： rollout，，。

，OPD ：

-  **distillation**， teacher。
-  **on-policy**， student 。
-  **offline**， student rollout。
-  off-policy ， teacher， teacher 。

## 

（Knowledge Distillation, KD）：，。： teacher ，student 。LLM  KD ： teacher logits， teacher ；、、、。[^kd_survey_xu][^kd_survey_yang]

。DeepSeek-R1 ：， SFT 。， RL 。

：** student  teacher ， student 。**

 prompt  $x$，teacher ：

$$y^{T} = (y_1^T, y_2^T, \dots, y_T^T) \sim q(\cdot \mid x)$$

：

$$\mathcal{L}_{\text{off-policy}}(\theta) = -\sum_t \log \pi_\theta(y_t^T \mid x, y_{<t}^T)$$

 $x, y_{<t}^T$  teacher。 student  3 ， $x, y_{<3}^{S}$。 teacher ，SFT ""。， exposure bias，。DAgger ：， learner 。[^dagger]

OPD  LLM 。

## Online OPD vs Offline OPD

 OPD  on-policy，[Lightning OPD](https://arxiv.org/html/2604.13010v1)  offline on-policy distillation？。"""teacher "。

** OPD  online 。**  student ，teacher  token  log-prob， student。 student ，、、。

```text
 student 
→ teacher 
→  student
→  student 
→ teacher 
→ ...
```

 on-policy ，：teacher  student ， OPD  live teacher server 。Lightning OPD  OPD ：teacher  rollout  log-prob。[^lightning_opd]

**Lightning OPD  offline 。**  SFT student， SFT student ；teacher ， token  log-prob 。 OPD ， teacher server， teacher ， student  log-prob。

```text
：
SFT student 
→ teacher 
→  token  teacher log-prob

：
 teacher log-prob
→  student log-prob
→  student
```

Lightning OPD  on-policy， teacher ， student 。teacher  student ， OPD ：**teacher  student **。

 Lightning OPD  online OPD， student ，rollout 。：OPD / RL  student  SFT ， SFT student  rollout  student  rollout。：**teacher consistency**—— SFT  teacher  OPD  teacher ，。[^lightning_opd]

：

|                     |                                   |                                    |                          |
| ----------------------- | ------------------------------------- | -------------------------------------- | ------------------------------------ |
|  online OPD         | ，              |  live teacher，                | 、、         |
| Lightning / offline OPD | ，， teacher server | rollout ， teacher consistency | 、、student  |
|  offline KD         | ， SFT                      |  teacher                       | 、 student     |

：** OPD  online on-policy distillation；Lightning OPD  OPD ； teacher SFT  off-policy / offline distillation。**

##  reverse KL

 KL 。

 KD  forward KL：

$$D_{\text{KL}}(q \| \pi_\theta)$$

 student  teacher 。：teacher  0.7、 0.2、 0.1，student 。， teacher  student ：、 token ，。

MiniLLM ： LLM  reverse KL：

$$D_{\text{KL}}(\pi_\theta \| q)$$

reverse KL  mode-seeking： student  teacher ， teacher 。MiniLLM  on-policy ， exposure bias。[^minillm]

 reverse KL ：** student 。**  token  student  0，student ，teacher 。 recipe " OPD"，：

1.  off-policy SFT / SeqKD ， student  teacher 。
2.  OPD  student 。

Thinking Machines Lab ： off-policy reasoning distillation， OPD 。2026  OPD ：OPD  teacher ， teacher  student  student 。[^rethinking_opd]

。DistiLLM  skew KL  adaptive off-policy ， teacher  student 。[^distillm]

##  teacher 

， OPD " teacher， student"。 OPD ， **teacher  student **。

，teacher  student  $c_t=(x,y_{<t})$ ， student  token ，。 teacher ， token  student  token ，reverse KL ：student  teacher  token，teacher  student  token 。，。

 "thinking-pattern consistency" ：teacher  student 。 teacher ， student ； reasoning teacher ， base student  prompt 。，， token 。

2026  OPD """"。[^rethinking_opd] teacher ，；teacher ， student 。 SFT ， SFT  student  teacher ； OPD ，student ，teacher  student 。"teacher ""teacher  student "。

**。**  teacher  pipeline  sibling， recipe ， student ，。 RL post-training、 teacher，， student 。

**OPD  teacher  student， teacher "" student。** "" teacher， student ； RL、、 teacher，。

 OPD ""。off-policy SFT  student  teacher ；OPD  student —— teacher ， teacher 。

### Overlap token 

 token ： student  teacher  overlap tokens ，； non-overlap tokens，。 top-$k$ OPD ， run  loss ，。

 token ——。 token ，teacher  log-prob ， student "，"。，teacher  student  token， student 。

** token 。**  student 、teacher 。overlap token ： student ， teacher 。non-overlap token ——， student 。

： teacher  reward / rollout，。**，。**  OPD run  reward 、teacher ，。

### 

OPD  token  reward，：，student  teacher ，teacher  log-prob 。：suffix ，。

 OPD  scaling bottleneck  teacher forward ， ****。，teacher  student ； 15K token 、、 agent ，teacher 。、、 horizon， overlap  token-level loss。

## ：OPD 

 OPD 。 OPD ：student ，teacher  student  token 。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

student_name = "Qwen/Qwen2.5-0.5B-Instruct"
teacher_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(student_name)
student = AutoModelForCausalLM.from_pretrained(
    student_name, torch_dtype=torch.bfloat16, device_map="auto"
)
teacher = AutoModelForCausalLM.from_pretrained(
    teacher_name, torch_dtype=torch.bfloat16, device_map="auto"
)
student.eval()
teacher.eval()

prompt = "Solve: if x + 3 = 7, what is x? Show your work."
inputs = tokenizer(prompt, return_tensors="pt").to(student.device)
prompt_len = inputs["input_ids"].shape[1]

# Step 1: student 
with torch.no_grad():
    output_ids = student.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.7,
    )

full_ids = output_ids

# Step 2:  student  teacher  token log-prob
def next_token_logps(model, input_ids):
    """ token  log probability。

    logps[:, i]  i  logits  token i+1  log-prob。
    """
    logits = model(input_ids).logits
    logps = torch.log_softmax(logits[:, :-1], dim=-1)
    next_ids = input_ids[:, 1:]
    return logps.gather(-1, next_ids.unsqueeze(-1)).squeeze(-1)

with torch.no_grad():
    student_logps = next_token_logps(student, full_ids)
    teacher_logps = next_token_logps(teacher, full_ids.to(teacher.device)).to(student_logps.device)

# Step 3:  token reward（teacher  - student ）
gen_mask = torch.zeros_like(student_logps, dtype=torch.bool)
gen_mask[:, prompt_len - 1 :] = True  # 

token_rewards = teacher_logps - student_logps
generated_ids = full_ids[:, 1:][gen_mask]
generated_rewards = token_rewards[gen_mask]

for tok_id, reward in zip(generated_ids[:32], generated_rewards[:32]):
    token = tokenizer.decode([tok_id.item()])
    print(f"{token!r:12s} reward={reward.item():+.3f}")
```

：

- `next_token_logps()`  student  teacher， log probability。 GRPO  log prob ——OPD  RL 。
- `token_rewards = teacher_logps - student_logps`： OPD  per-token reward。 teacher  student  token， teacher 。
- ""。，：batch rollout、reward /、policy gradient 。

：

```python
for prompts in dataloader:
    # Step 1: student rollout
    trajectories = student.rollout(prompts)
    student_logps = student.logprobs(trajectories)
    teacher_logps = teacher.logprobs(trajectories)

    # Step 2:  per-token advantage
    advantages = teacher_logps - student_logps
    advantages = normalize_and_mask(advantages, trajectories.response_mask)

    # Step 3: 
    loss = policy_gradient_loss(
        new_logps=student.logprobs(trajectories),
        old_logps=student_logps.detach(),
        advantages=advantages.detach(),
    )
    loss.backward()
    optimizer.step()
```

 KL  reference、、、prompt  eval gating。OPD ""， teacher log-prob  RL 。

## OPD 

###  SFT / SeqKD 

SFT 。、、 student 。OPD  SFT， SFT  student 。

：

- SFT / SeqKD： teacher 
- OPD：student ，teacher 

###  GRPO / RLVR 

GRPO / RLVR ：、、。，。 2000-token ， 0  1。

OPD  teacher， token 。， RM， teacher 。

：， teacher log-prob  shaping 。Thinking Machines Lab " per-token reward + sequence-level environment reward"。[^tml_opd]

###  DPO 

DPO ：。OPD ： teacher  token reward model。

 DPO ， chosen / rejected ；OPD ， teacher、。

### 

 OPD  teacher log-prob， logprob-access teacher。 teacher  API ， logits。 OPD ： 2025  GAD  student  generator， discriminator  teacher  student ， discriminator  student  on-policy reward model。[^blackbox_opd]

，： hack、、。

 teacher-free ：Self-Distilled Reasoner  teacher  student， teacher 。[^self_distilled_reasoner]

##  OPD

OPD ：

|                      |  OPD                                     |
| ------------------------ | -------------------------------------------------- |
|  | ，teacher              |
|            | ， teacher  token        |
|            |  teacher  instruction following  |
|  SFT   | student  teacher  token          |
|  RL            | teacher forward  RL      |

：

|                             |                                           |
| ------------------------------- | --------------------------------------------- |
|  teacher              | OPD  teacher ， |
| student ， token  | reverse KL            |
| teacher  student  |  token ，         |
|  token reward   |                 |
|  teacher API            |  reward/discriminator             |
| teacher  pipeline   | ， student        |

2026  OPD survey ：（logit-based、outcome-based、self-play）、teacher access（white-box、black-box、teacher-free） loss granularity（token-level、sequence-level、hybrid）。[^opd_survey] ："OPD "，，。

## 

###  recipe

 OPD ，：

**， teacher。** teacher ， student 、 student ， student  tokenizer、、。，""。： pipeline  teacher vs  RL /  teacher。， OPD ，。

**， off-policy 。**  teacher ， SFT student。， student  teacher 。 overlap ratio ， OPD ； teacher rollout  SFT， student ， on-policy。

**， prompt 。** prompt ， student 。 teacher 、， OPD  overlap； OOD prompt， student  teacher 。

**， student rollout。**  prompt  2-8 ， token、log-prob、mask、、。 PPO / GRPO  rollout 。

**，teacher 。**  student  teacher forward， token  log-prob。 teacher  logits； teacher  reward 。 teacher ，： student， student ， teacher 。

**， student。**  teacher_logp - student_logp  per-token advantage， PPO-style loss  importance-sampling loss。 entropy、KL、response length ， student 。 token、 overlap token  loss， 10K token 。

**，。** ，。 final reward ， OPD reward  token-level shaping。 OPD ：teacher  token，。

**， eval gating。** OPD  teacher  student。 benchmark，、、、。

### 

 OPD，。：

1. teacher  token-level ？
2. offline ？
3. online rollout  offline ？

#### 0.  teacher ""

 sanity check，。： teacher  student 。

 50-100  prompt， student  2-4 。teacher ， student  token  log-prob。：teacher  token， student ？teacher ，、？、， student token ， teacher  student 。

 insight：OPD  teacher""， teacher  student 。 reward， token ，。

#### 1. Lightning OPD smoke test

 offline ，、、。

：

-  prompt：200-1000 
-  prompt：50-200 
- student：0.5B-1.5B 
- teacher：，
- ：LoRA ，100-500 steps 

：

```text
1.  SFT student 
2.  teacher  token  log-prob
3.  student， teacher  token 
4.  held-out prompt 
5. ：、、、teacher score
```

"teacher score "，：

|             |                                      |
| ----------------- | ---------------------------------------- |
| held-out  | ，                   |
|       | ，                 |
|             |                                    |
| teacher score     | ，、   |
|           |  20 ， teacher |

 teacher score ， student  teacher ，。 mask、、prompt  teacher consistency。

#### 2. Online OPD 

 offline smoke test ， online 。 2-3 ：

```text
Round 1:  student  rollout → teacher  →  50-100 steps
Round 2:  student  rollout → teacher  → 
Round 3: ，
```

 Lightning OPD： prompt、 teacher、， rollout 。 online  offline。

| ...               |                                            |
| --------------------------- | ---------------------------------------------- |
| offline  online     |  Lightning OPD ， live teacher |
| online              | student ， rollout       |
| online ，offline  | teacher ， online  |
|                 |  SFT 、teacher       |

： offline ， Lightning； online ， teacher server。

#### 3. 

 OPD，：

|               |                                              |
| ----------------- | ------------------------------------------------ |
| student / teacher | 、、                         |
|               | prompt 、、                |
| rollout           | 、temperature、max tokens              |
| reward            | teacher log-prob 、        |
|               | online  offline、steps、LoRA rank            |
|               | 、、、                 |
| insight       | teacher ，student  |
|               |  online、 Lightning、 SFT        |

 loss 。OPD  loss  student  teacher，。

## 

 OPD  OPSD 。。

### 

|  |                                                                                                                                                                   |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| OPD  | On-Policy Distillation， rollout，**** token 。                                                                                 |
| OPSD | On-Policy Self-Distillation，**** teacher  student，teacher  privileged context（ ground-truth 、""）。 |

### OPD 

#### 1. slime（THUDM）

slime  GLM-4.5 / 4.6 / 4.7 ，OPD ，** RL  KL **。

**：**

- **SGLang **：teacher  SGLang server ， `--rm_url`  rollout  token-level log-probs。 teacher 。
- **Megatron **：teacher  Megatron ， `--opd-teacher-load`  checkpoint。 teacher  policy/ref 。

**（Megatron ）：**

```bash
python train.py \
  --use-opd \
  --opd-type megatron \
  --opd-kl-coef 1.0 \
  --opd-teacher-load /path/to/teacher_ckpt \
  --adv_estimator grpo   #  ppo / reinforce_plus_plus
```

**：**

- OPD  advantage estimator ， advantage  reverse KL 。
- `slime/rollout/on_policy_distillation.py`  SGLang  reward_func： sample  teacher server，trim  response span  Sample。
-  Qwen3-8B student + Qwen3-32B teacher， DAPO-Math-17k  Math500  76%  94%。

** OPSD**：README  "use a different (stronger) model as the teacher"， privileged context 。

---

#### 2. veRL（ByteDance Seed）

veRL  RL ，OPD  trainer 。

**：** `examples/on_policy_distillation_trainer/`

**：**

```bash
bash examples/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh
```

**（Hydra YAML）：**

```yaml
distillation:
  enabled: True
  teacher_models:
    teacher_model:
      model_path: 'Qwen/Qwen3-32B' # HF 
  distillation_loss:
    loss_mode: 'k3' #  k1 / k3 / forward_kl_topk 
    use_policy_gradient: True #  GRPO PG loss 
    topk: 64 # teacher  top-k logits，
```

**：**

- teacher ** Ray ** serving， student 。
-  `topk` sparse logits：teacher  top-64  logit ，student  KL， logits。
-  FSDP  Megatron ， vLLM 。
-  VLM（`run_qwen3_vl_8b_fsdp.sh`） OPD 。

** OPSD**： `on_policy_distillation_trainer`  `teacher_models` 。 `HJSang/OPSD_OnPolicyDistillation`  veRL ， README  **"TODO: Add OPSD support. Currently only OPD is included"**。

---

#### 3. NeMo RL（NVIDIA）

NeMo RL  NVIDIA ，OPD 。

**：** `nemo_rl/algorithms/distillation.py`

**：**

```bash
python examples/run_distillation_math.py
```

**（YAML）：**

```yaml
teacher:
  model_path: 'nvidia/Nemotron-4-340B' # teacher 
  tensor_parallel_size: 4 # teacher  TP
distillation:
  topk_logits_k: 64 # sparse top-k teacher logits
loss_fn:
  kl_type: 'reverse' # forward / reverse / mixed
```

**：**

- ****：Phase 1  teacher  micro-batch  logits， CPU  teacher；Phase 2  student  forward + backward。 teacher  student 。
- ****：teacher  student  Tensor Parallelism、Context Parallelism 。
-  multi-turn rollout， `max_rollout_turns` 。
-  NeMo ， Megatron 。

** OPSD**：`teacher: PolicyConfig` ， privileged context 。

---

#### 4. TRL（HuggingFace）

TRL  OPD trainer ，**、**。

**：** `trl/experimental/gkd/`

**：**

```python
from trl import GKDTrainer, GKDConfig

trainer = GKDTrainer(
    model="Qwen/Qwen3-8B",           # student
    teacher_model="Qwen/Qwen3-32B",  # teacher（）
    args=GKDConfig(
        kl_type="reverse",           # "forward" / "reverse" / "jsd"
        temperature=1.0,
        per_device_train_batch_size=4,
    ),
    train_dataset=dataset,
)
trainer.train()
```

**：**

- `GKDTrainer`  `DPOTrainer`，。
-  forward KL、reverse KL、JSD 。
-  Accelerate，、DeepSpeed、FSDP。
-  `minillm/`、`gold/`、`online_dpo/` ， OPD 。

**TRL ：**

- **ms-swift**：`examples/train/rlhf/gkd/`  TRL `GKDTrainer`， Megatron 。
- **LLaMA-Factory**： TRL  OPD，。

---

#### 5.  OPD 

|                              |             |                                         |                                                                  |
| -------------------------------- | --------------- | ----------------------------------------------- | -------------------------------------------------------------------- |
| **rLLM**（UC Berkeley Sky）      |  OPD + OPSD | `rllm/trainer/distill/`                         |  GPU  tinker， GPU  verl 。AwesomeOPD  OPSD 。 |
| **AReaL**（AntGroup / Tsinghua） |  RL   | `examples/distillation/gsm8k_grpo_distill.yaml` | 。                                             |
| **ROLL**（Alibaba）              |  RL   | `roll/pipeline/distill/`                        |  VLM， divergence 。                               |
| **SkyRL**（UC Berkeley NovaSky） | RL      | `skyrl-train/examples/on_policy_distillation/`  | NovaSky ， Sky 。                      |
| **KDFlow**（BJTU）               | KD-first    | `examples/on_policy_kd/`                        | SGLang teacher + FSDP2 student ， tokenizer  VLM。   |

---

#### 6.  OPD 

**OpenRLHF**  AwesomeOPD 。 rollout / teacher / update ，：

- teacher  remote Ray worker， full logits 。
-  on-policy distillation ，。
-  OPD 。

---

### OPSD 

OPSD（On-Policy Self-Distillation） OPD ：**** teacher  student，teacher  privileged context（ground-truth 、、）， rollout  token 。

#### 1. TRL（HuggingFace）—— 

**：** `trl/experimental/self_distillation/`

**：**

- `SelfDistillationMixin._split_prompt_and_privileged_context()`  batch  `prompt`  `privileged_context`。
-  model  forward：
  - student forward：`prompt + completion`
  - teacher forward：`prompt + privileged_context + completion`（teacher ）
-  reverse KL：`KL(teacher || student)`， loss。

**：**

- `BaseSelfDistillationTrainer`：， vLLM rollout。
- `SelfDistillationMixin`： loss ， `grpo`、`bnpo`、`dr_grpo`、`dapo`  loss type。
- `SDPO`（Self-Distillation Policy Optimization）： trainer 。

**：**
 `prompt`  `privileged_context` 。，`privileged_context`  ground-truth 。

**：**

```python
from trl import SDPOTrainer, SelfDistillationConfig

trainer = SDPOTrainer(
    model="Qwen/Qwen3-8B",
    args=SelfDistillationConfig(
        kl_type="reverse",
        loss_type="grpo",      #  bnpo / dapo 
    ),
    train_dataset=dataset,     #  "prompt"  "privileged_context"
)
trainer.train()
```

**：**

- `alpha=0`：reverse KL（$D_{KL}(teacher || student)$）
- `alpha=1`：forward KL（$D_{KL}(student || teacher)$）
- `0 < alpha < 1`：JSD 

**：** ，API ；， teacher。

---

#### 2. rLLM（UC Berkeley Sky）

AwesomeOPD  rLLM  `examples/math_distill/`  OPSD （ `opsd/` ）。，：

-  GPU （tinker ）
-  GPU （verl ）

 GitHub ，。

---

#### 3.  OPSD 

|                          |                                                                         |
| ---------------------------- | --------------------------------------------------------------------------- |
| **slime**                    |  `--opd-teacher-load` ， privileged context 。  |
| **veRL**                     |  `teacher_models.teacher_model.model_path` 。     |
| **NeMo RL**                  | `teacher: PolicyConfig` ，。        |
| **ms-swift / LLaMA-Factory** |  TRL GKDTrainer  OPD，TRL  self_distillation 。 |
| **OpenRLHF**                 |  OPD， OPSD。                                                     |

### 

- ** OPD + **（SGLang / Megatron / vLLM teacher server）： **slime**、**veRL**  **NeMo RL**。，slime  veRL 。
- ** OPSD（）**： **TRL** 。 TRL ， `self_distillation/` 。
- ** SWIFT / ModelScope  LLaMA-Factory **：OPD  TRL GKDTrainer ， OPSD 。
- ** OPD **：**TRL**  `GKDTrainer`  **veRL**  `on_policy_distillation_trainer` 。

## 

OPD ：

- off-policy  token ， student 。
- RL  on-policy，，。
- OPD ：student ，teacher  token 。

、。 OPD  RL ，—— teacher，，" student "。

2026  insight， OPD "teacher ""teacher "。； teacher  student 、。 pipeline ：， teacher-aligned prompt  teacher ，。

，DPO、GRPO、RLVR  OPD ：** RLHF ，？** DPO ，RLVR ，OPD  teacher。，。

## 

[^kd_survey_xu]: Xu X, Li M, Tao C, et al. [A Survey on Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2402.13116), arXiv 2024.（ algorithm、skill、verticalization  LLM KD）

[^kd_survey_yang]: Yang C, Lu W, Zhu Y, et al. [Survey on Knowledge Distillation for Large Language Models: Methods, Evaluation, and Application](https://arxiv.org/abs/2407.01885), arXiv 2024.（ white-box / black-box KD、）

[^opd_survey]: Song M, Zheng M. [A Survey of On-Policy Distillation for Large Language Models](https://arxiv.org/abs/2604.00626), arXiv 2026.（ OPD  f-divergence ，、teacher access、loss ）

[^dagger]: Ross S, Gordon G, Bagnell D. [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html), AISTATS 2011.（DAgger： learner ）

[^gkd]: Agarwal R, Vieillard N, Zhou Y, et al. [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649), ICLR 2024.（GKD： student  teacher ）

[^minillm]: Gu Y, Dong L, Wei F, Huang M. [MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543), ICLR 2024.（ reverse KL  on-policy  LLM ）

[^distillm]: Ko J, Kim S, Chen T, Yun S. [DistiLLM: Towards Streamlined Distillation for Large Language Models](https://proceedings.mlr.press/v235/ko24c.html), ICML 2024.（ skew KL  adaptive off-policy  LLM ）

[^tml_opd]: Lu K, Thinking Machines Lab. [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/), 2025.（ OPD  Tinker ， Qwen3  personalization ）

[^rethinking_opd]: Li Y, Zuo Y, He B, et al. [Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe](https://arxiv.org/abs/2604.13016), arXiv 2026.（ OPD 、token-level ）

[^lightning_opd]: Shi Z, Zhang J, Jiang W, et al. [Lightning OPD: Cost-effective On-Policy Distillation](https://arxiv.org/html/2604.13010v1), arXiv 2026.（ OPD ： teacher log-prob， live teacher server）

[^self_distilled_reasoner]: Zhao S, Xie Z, Liu M, et al. [Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models](https://arxiv.org/abs/2601.18734), arXiv 2026.（ teacher  student）

[^blackbox_opd]: Ye T, Dong L, Chi Z, et al. [Black-Box On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2511.10643), arXiv 2025.（GAD： teacher logits ， discriminator  on-policy ）
