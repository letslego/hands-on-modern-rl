# 9.4 DeepSeek-R1  DAPO—— RL 

 GRPO —— Critic，。， 2025  RL ：DeepSeek-R1-Zero  RL  SFT，DAPO  GRPO 。

## DeepSeek-R1-Zero

 DeepSeek-R1 ，：**Base  SFT（）， RL**。—— SFT  RL，、""，RL 。

2025  1 ，DeepSeek  AI 。：**（、）， SFT **。 Base  GRPO 。 DeepSeek-R1-Zero—— SFT  RL 。

？ RLHF ，SFT  RL 。：Base ""，""。 Base  RL，，RM 。 SFT ， RL 。

 DeepSeek-R1-Zero 。（，），""——，。，，RL 。，""。

### 

R1-Zero **（Emergence）**。，：

- **（Chain-of-Thought）**："""、、"，
- ****：，，
- ****：，

，""。DeepSeek  **"Aha Moment"（）**——，""，。

，DeepSeek ：

- ****（0-100 ）：，
- ****（100-500 ）：，
- ****（ 500-1000 ）：，"，"
- ****（1000+ ）："→→"

：**？** ：""（、、），RL ""。 1-Shot RLVR ——，RL ""。

### ：SimpleRL-reason

，：R1-Zero ， DeepSeek ？ base model，，" SFT、 RL "？

[SimpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason)  [SimpleRL-Zoo](https://arxiv.org/abs/2503.18892) 。， R1-Zero ： base model ， SFT， Reward Model，。

：

```text
 x
→ base model  y
→  y 
→ 
→  0/1 
→  PPO / RL 
```

 R1-Zero ：， RM，。SimpleRL-reason  OpenRLHF， Ray 、vLLM ； SimpleRL-Zoo ， Llama、Mistral、DeepSeek-Math、Qwen2.5-Math  Qwen2.5 。

，" RL "：

|                          | SimpleRL-Zoo                               |
| ---------------------------- | -------------------------------------------------------- |
| base model ？    |  zero RL                 |
| ？       | 、             |
| ？ | 、                 |
| ""？     | ，、 |

，SimpleRL-reason  R1-Zero ，。：zero RL ，； base model ，，RL 。

。SimpleRL-reason "simple"，。、 rollout 。， R1-Zero 、，。

### R1-Zero 

 R1-Zero  RL ，：****。 SFT，（）、、。，""。

， DeepSeek-R1 ：

1. ****： SFT （）
2. ** GRPO**：（）
3. ****： GRPO 
4. **SFT **：
5. ** RL**： RM  GRPO 

## DAPO

GRPO " Critic  RL"，。DAPO（Decoupled Clip and Dynamic Sampling Policy Optimization）， NeurIPS 2025  poster。

### DAPO 

|                         | GRPO                              | DAPO                             |               |
| --------------------------- | --------------------------------------- | -------------------------------------- | ----------------- |
| **Clip-Higher**             | ，      | ， |         |
| ****                |  prompt ，        |  prompt            |  2-3x |
| **Token **            |  reward ， token  | Token ，     |   |
| **Overlong Reward Shaping** | ，  |                      |         |

**Clip-Higher** ：GRPO （ $[0.8, 1.2]$），。（ 0.01），0.8  0.008——。DAPO ，。

****""。，（），。DAPO ""， prompt。 AIME 2024 ，DAPO  DeepSeek-R1 **** 50 。

**Token ** GRPO 。 GRPO ：（），（）。，， 80% ，。Token  GRPO " token 、"，。[ 7  GAE](../chapter07_ppo/gae-reward-model)——， token 。

**Overlong Reward Shaping**  GRPO ：。""（）， 2000+ token 。GRPO ，。—— 499 token ，501 token ——。DAPO ，。

```python
# ==========================================
# DAPO 
# ==========================================
def dynamic_sampling(prompts, model, reward_fn, threshold=0.95):
    """
     prompt
    """
    useful_prompts = []

    for prompt in prompts:
        #  prompt ，
        correct_count = 0
        num_samples = 8
        for _ in range(num_samples):
            response = model.generate(prompt)
            reward = reward_fn(prompt, response)
            if reward >= 1.0:  # 
                correct_count += 1

        accuracy = correct_count / num_samples
        #  prompt（）
        if accuracy < threshold:
            useful_prompts.append(prompt)

    print(f": {len(prompts)} ")
    print(f": {len(useful_prompts)} ")
    print(f": {len(prompts) - len(useful_prompts)} （）")
    return useful_prompts
```

DeepSeek-R1-Zero  DAPO  RL —— SFT， Critic，。：**？**  RLVR—— Reward Model。
