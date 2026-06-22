# B.4  RL 

> ，——`actor/pg_clipfrac`、`critic/advantages/mean`、`timing_s/gen`、`perf/mfu/actor`……，？
>
> 。：****；****；****；****。——。
>
> ""。 veRL、OpenRLHF、TRL ，（）。

---

## 1.  — ""

、，：**？**

。，——，。

|                       |                                                                     |
| ------------------------- | ----------------------------------------------------------------------------- |
| `val-core/*/acc/mean@1`   | ——。                            |
| `val-aux/*/reward/mean@1` | 。（ GSM8K ）， |
| `val-aux/num_turns/*`     |                                                         |

 `val-core` （），`val-aux` （、）。 `val-core`， `val-aux`。

****：，`val-core` 。，——（ 2 ），（ 4 ）。

---

## 2. Actor  — ""

（），****。Actor ：、、——。

### 2.1 loss：""""

|                  |                                                                       |
| -------------------- | ------------------------------------------------------------------------------- |
| `actor/loss`         | Actor ， `pg_loss + `。     |
| `actor/pg_loss`      | （policy gradient）。"" |
| `actor/entropy_loss` | 。""                                    |
| `actor/kl_loss`      | KL （ `use_kl_loss=True` ）。 |
| `actor/kl_coef`      | KL                                                                |

****：`actor/loss` 。，、、。`pg_loss` （），。

### 2.2 ：

|               |                                        |
| ----------------- | ------------------------------------------------ |
| `actor/grad_norm` |  L2 。"" |

****：，。`grad_norm` ——（），（）。 `grad_norm` 。 10 ，。

### 2.3 ：""

|             |            |
| --------------- | -------------------- |
| `actor/entropy` |  |

：（），；（），。

- **，**：，
- **，**：
- ****：""，， reward hacking（""）
- ****：

### 2.4 KL ：

|            |                                                  |
| -------------- | ---------------------------------------------------------- |
| `actor/ppo_kl` | （） KL  |

。KL ""。，。

****：KL 、。（ 0.02  0.15），——。 PPO  KL （ 0.01~0.1）， KL 。

### 2.5 Clip Fraction：""

|                       |                                            |
| ------------------------- | ---------------------------------------------------- |
| `actor/pg_clipfrac`       | PPO —— |
| `actor/pg_clipfrac_lower` |                                        |

PPO （clipping），——，。clipfrac ""。

****： 0.1~0.3。 0.3， clip —— batch size。 clipfrac  0，。

### 2.6 

|        |           |
| ---------- | ------------------- |
| `actor/lr` |  actor  |

****： schedule （ warmup + cosine decay）。，，。

---

## 3. Critic /  — ""

 RL ：Actor （），Critic （""）。，PPO  GRPO 。

### PPO ：""

PPO  Critic ，。

|                      |                                                                                                |
| ------------------------ | -------------------------------------------------------------------------------------------------------- |
| `critic/v_loss`          | Critic ——                                                                        |
| `critic/vpred/mean`      | Critic  value                                                                                  |
| `critic/vpred/var`       | Critic                                                                                         |
| `critic/score/mean`      |                                                                                        |
| `critic/rewards/mean`    | ——                                                                   |
| `critic/advantages/mean` | （ GAE ）。 Actor ——"，"，"，" |
| `critic/returns/mean`    |                                                                                                  |

****：`critic/v_loss` （）。`critic/rewards/mean` （）。 rewards  `v_loss` ， Critic  Actor ， Critic 。

### GRPO ：""，""

GRPO  Critic （ `Disabled critic as algorithm.adv_estimator != gae`）。 `critic/rewards/mean`、`critic/advantages/mean` —— batch ，，** Critic **。

---

## 4.  — ""

，。

：，，；，""。

### 

|                             |                                                   |
| ------------------------------- | ----------------------------------------------------------- |
| `response_length/mean`          |  token                                      |
| `response_length/max`           |                                                     |
| `response_length/min`           |                                                     |
| `response_length/clip_ratio`    |  `max_response_length` —— |
| `response_length_non_aborted/*` |                                     |
| `response/aborted_ratio`        |                                 |

### Prompt 

|                        |                                                     |
| -------------------------- | ------------------------------------------------------------- |
| `prompt_length/mean`       |  prompt  token                                    |
| `prompt_length/max`        |  prompt                                                   |
| `prompt_length/min`        |  prompt                                                   |
| `prompt_length/clip_ratio` |  prompt  `max_prompt_length` —— |

### 

- **`clip_ratio` **： prompt 。 response， `max_response_length`； prompt，
- **`response_length/mean`  + reward **： reward hacking——""，
- **`response/aborted_ratio` **：。， EOS 

---

## 5. DPO / KTO / SimPO  — ""

 PPO  GRPO——（on-policy）。DPO ：" vs "，。。

### DPO 

|                |                                           |
| ------------------ | --------------------------------------------------- |
| `loss/dpo`         | DPO 。                          |
| `rewards/chosen`   | ""。            |
| `rewards/rejected` | ""。 chosen     |
| `rewards/margins`  | chosen  rejected —— |
| `rewards/accuracy` |  chosen                           |
| `logps/chosen`     |  chosen  log                            |
| `logps/rejected`   |  rejected  log                          |

****：`rewards/margins`  DPO ——。。、，****——，。

### KTO  SimPO

- **KTO**：，""""。 `kl_estimate`（） `loss/kl`（KL ）
- **SimPO**：， DPO  reference ， length-normalized reward 

---

## 6. Reward Model  — ""

 PPO ， Reward Model（RM）——""，。RM 。

|                       |                                     |
| ------------------------- | --------------------------------------------- |
| `rm/loss`                 | RM （）。     |
| `rm/accuracy`             | RM ""             |
| `rm/reward_margin`        | RM ——"" |
| `rm/reward_chosen/mean`   | RM                            |
| `rm/reward_rejected/mean` | RM 。 chosen      |
| `rm/grad_norm`            | RM                                  |

###  RM 

：****（accuracy ）****（margin ，、）。

，：

- **accuracy ， PPO **：RM ，
- **margin  0**：RM ""，
- ** RM ， RM **：RM ""—— RM ，RM ""

---

## 7.  — ""

|                   |            |
| --------------------- | -------------------- |
| `num_turns/min`       |  |
| `num_turns/max`       |  |
| `num_turns/mean`      |          |
| `val-aux/num_turns/*` |    |

****： GSM8K ， 2 （1  user + 1  assistant）。 Agentic RL（），。 `num_turns/max` ，。

---

## 8.  — ""

""。，""。

`global_seqlen/*` **，**。 GPU（partition / rank） token ，。

|                          |                                   |
| ---------------------------- | ------------------------------------------- |
| `global_seqlen/min`          | ， token        |
| `global_seqlen/max`          | ， token        |
| `global_seqlen/minmax_diff`  | `max - min`。， |
| `global_seqlen/balanced_min` |                       |
| `global_seqlen/balanced_max` |                       |
| `global_seqlen/mean`         |                           |

****：`minmax_diff` 。： 80% ，，。——`minmax_diff` ，，。

---

## 9.  — ""

|                       |                                                            |
| ------------------------- | -------------------------------------------------------------------- |
| `timing_s/step`           |  step                                                    |
| `timing_s/gen`            | 。                                   |
| `timing_s/reward`         |  reward 。（）， |
| `timing_s/old_log_prob`   |  log probability  entropy                    |
| `timing_s/ref`            |  log probability                               |
| `timing_s/adv`            |                                                  |
| `timing_s/update_actor`   |  Actor                                             |
| `timing_s/update_critic`  |  Critic （PPO ，GRPO ）                    |
| `timing_s/update_weights` |                                |

****： `timing_s/step` ，。 `timing_s/gen`（）。，。

### 

 Agentic ，：

- `timing_s/agent_loop/generate_sequences/min|max|mean`：
- `timing_s/agent_loop/tool_calls/min|max|mean`：
- `timing_s/agent_loop/slowest/*`：

###  token 

`timing_per_token_ms/*`  token ，" token "。** run **—— PPO vs GRPO ， `timing_s/gen` （ run  token）。

：`timing_per_token_ms/gen`、`timing_per_token_ms/ref`、`timing_per_token_ms/adv`、`timing_per_token_ms/update_actor`。

---

## 10.  — "GPU "

|                     |                                   |
| ----------------------- | ------------------------------------------- |
| `perf/total_num_tokens` |  token                  |
| `perf/time_per_step`    |                                 |
| `perf/throughput`       |  token——    |
| `perf/mfu/actor_infer`  | Actor  MFU（ FLOPs ） |
| `perf/mfu/actor`        | Actor  MFU                  |

### MFU 

MFU "GPU "。—— 100 / 40 ， 40%。

 MFU  50%~60% 。 30%  GPU （、、），。

---

## 11. 

。，""。

|        | veRL / OpenRLHF        | TRL                             |
| ---------- | ---------------------- | ------------------------------- |
|      | `actor/entropy`        | `policy/approx_kl`  `entropy` |
| KL     | `actor/ppo_kl`         | `objective/kl`                  |
|    | `actor/grad_norm`      | `loss/total`        |
|    | `response_length/mean` | `response_length`               |
|    | `timing_s/gen`         |                   |
|    | `critic/rewards/mean`  | `rewards`                       |
| DPO  | `rewards/margins`      | `rewards/margins`（）       |

 TRL  `SFTTrainer` + `DPOTrainer`， wandb / tensorboard， TRL  Logging 。

---

## 12. ：

### PPO / GRPO 

1. ****：`val-core/*/acc/mean@1`（）、`critic/rewards/mean`（）
2. ****：`actor/loss`、`actor/grad_norm`、`actor/ppo_kl`、`actor/pg_clipfrac`
3. ****：`perf/throughput`、`timing_s/gen`
4. ****：`response_length/mean`、`response_length/clip_ratio`、`global_seqlen/minmax_diff`

### DPO / KTO 

1. ****：`rewards/margins`（）、`rewards/accuracy`（）
2. ****：`loss/dpo`（）
3. ****：`rewards/accuracy`  99%+  → 

### Reward Model 

1. ****：`rm/accuracy`（）、`rm/reward_margin`（）
2. ****： accuracy 

### 

|                  |                             |                                            |
| -------------------- | ------------------------------- | ---------------------------------------------- |
| Reward           | `actor/entropy`         | `actor/pg_clipfrac`                    |
| KL               | `actor/ppo_kl`              |  schedule                        |
|  reward  | `response_length/mean`  | `response/aborted_ratio`               |
|              | `timing_s/step`           | `perf/throughput`、`global_seqlen/minmax_diff` |
|  OOM             | `global_seqlen/max`     | `response_length/max`、`prompt_length/max`     |
