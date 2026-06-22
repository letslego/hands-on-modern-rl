# Part 1:  — 

##  Part ？

，。 1  Stable Baselines3  CartPole —— "Hello World"。 2  TRL  Qwen2.5-0.5B  DPO ，。

，：

- ****：" →  →  → "。，。
- ****： $\pi(a|s)$ "，"。。
- ****： $\nabla_\theta J \approx \nabla_\theta \log \pi_\theta(a|s) \cdot G$ —— $G$ ，。
- **DPO **：$\mathcal{L}_{\text{DPO}} = -\log \sigma(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)})$。，。
- ****： RL  Gymnasium + Stable Baselines3； Transformers + TRL。

。

##  1 ：CartPole—— RL 

### Agent-Environment 

（Agent）（Environment）。：（），（），（），。，——。

 CartPole ， $s$  4 ：、。——， $\mathcal{A} = \{0, 1\}$。， +1 ；，。

：

```python
obs, info = env.reset()
while True:
    action = model.predict(obs)        # ：
    obs, reward, done, truncated, info = env.step(action)  # 
    if done or truncated:
        break
```

" →  →  → "。——DQN、PPO  DPO——： $\pi(a|s)$ ？

### 

 $\pi_\theta(a|s)$ 。 CartPole ， 64 ， 4 ， 2  logits， softmax 。

****。：（ $G_t$ ），；。

？ $J(\theta) = \mathbb{E}_{\pi_\theta}[G_t]$。$\theta$ ， $\pi_\theta(a|s)$——。 $J$  $\theta$ ， $\sum_a \nabla_\theta \pi_\theta(a|s) \cdot G_t$。：$\nabla \pi$ ，——。

：$\nabla \pi = \pi \cdot \nabla \log \pi$（）。 $\pi$ ，：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]$$

。 $(s_t, a_t, G_t)$，****：

$$\nabla_\theta J(\theta) \approx \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$$

：$\nabla_\theta \log \pi_\theta(a_t|s_t)$ ""，$G_t$ ""——，。$\log$ ：（）， $\nabla \log \pi = \nabla \pi / \pi$ ，。 [ 5 ](../chapter05_policy_gradient/reinforce)。

 $\mathcal{L} = -\log \pi_\theta(a_t|s_t) \cdot G_t$。——**，。 REINFORCE 。

，""：（Reward），；（Entropy）$\mathcal{H} = -\sum_a \pi(a|s) \log \pi(a|s)$ ，。 RL 。

###  Stable Baselines3 

，：

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# 
env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=1)

# ——
model.learn(total_timesteps=80000)

# 
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f": {mean_reward:.2f} +/- {std_reward:.2f}")
```

 PPO ， 7 。：PPO ****，。

##  2 ：DPO——""

### 

。****（Pre-training），""，""。****（SFT），，""。****（RL Alignment），，""。

DPO（Direct Preference Optimization）。

### DPO 

DPO ： RL （ RLHF），。

 $(x, y_w, y_l)$， $x$ ，$y_w$ ，$y_l$ 。DPO ****：

$$r(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$

 $\pi_\theta$ ，$\pi_{\text{ref}}$ （）。$\beta$ ，——$\beta$ ，。

""？ RLHF 。DPO ，——（$\pi_\theta > \pi_{\text{ref}}$），；（$\pi_\theta < \pi_{\text{ref}}$），。

，DPO ：

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

 $\sigma$  sigmoid 。，：，。，$\sigma$ ，$\log \sigma$  0，。，，，。

###  TRL  DPO

， HuggingFace  TRL ，：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

# ——，
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
ref_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# 
training_args = DPOConfig(
    output_dir="./dpo_output",
    per_device_train_batch_size=2,
    learning_rate=1e-5,
    num_train_epochs=3,
    beta=0.1,       # KL 
)

# 
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=preference_dataset,  #  prompt, chosen, rejected 
    processing_class=tokenizer,
)
trainer.train()
```

：****，；****（Reward Margin，），。

## 

，。， Agent-Environment ——，，。，RL （CartPole），（DPO），：，。

 Part 2 ，——MDP、、DQN、、PPO—— LLM 。

> ****：[Part 2: ](/chapter03_mdp/intro)
