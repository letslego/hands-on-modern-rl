# 1：RL——CartPole 

> ****： RL ，。。

> 📁 ****：[1-ppo_cartpole.py](https://github.com/letslego/hands-on-modern-rl/blob/main/code/chapter01_cartpole/1-ppo_cartpole.py) · [2-pytorch_ppo.py](https://github.com/letslego/hands-on-modern-rl/blob/main/code/chapter01_cartpole/2-pytorch_ppo.py) · [requirements.txt](https://github.com/letslego/hands-on-modern-rl/blob/main/code/chapter01_cartpole/requirements.txt)

## 1.1 ： CartPole 

，。：（Agent）（Environment），，。

，""？——CartPole（）。 `print("Hello World")` ，，。

![CartPole ](./images/cartpole-real-env-frames.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em> 1-1：CartPole ：、。，，。</em>
</div>

：？

，，（ Intel Mac、Apple Silicon、Windows/Linux）：

- ** (GPU)**：，CPU 。
- ****： 100MB~200MB 。
- ****：（`MlpPolicy`）， 64 ，。

， Gymnasium（ RL ）， Stable Baselines3 (SB3) 。 PyTorch ， SB3 ， PPO 。

。， CartPole 。

![PPO  CartPole ](./images/rl-training-loop.svg)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em> 1-2：PPO  CartPole 。 30 。</em>
</div>

### **：**

，，：

```bash
pip install "gymnasium[classic-control]" stable-baselines3
```

> ****：`stable-baselines3`  `PyTorch`。 PyTorch ，。。

### **：**

：

```bash
pip install -r requirements.txt
```

 CartPole ，****：

- [1-ppo_cartpole.py](https://github.com/letslego/hands-on-modern-rl/blob/main/code/chapter01_cartpole/1-ppo_cartpole.py)： SB3  PPO ，。
- [2-pytorch_ppo.py](https://github.com/letslego/hands-on-modern-rl/blob/main/code/chapter01_cartpole/2-pytorch_ppo.py)： PyTorch  PPO ，。

 SwanLab， `--gui` ：

```bash
#  A：SB3 （）
python 1-ppo_cartpole.py
python 1-ppo_cartpole.py --gui

#  B： PyTorch （）
python 2-pytorch_ppo.py
python 2-pytorch_ppo.py --gui
```

， `output/` 。

 `--gui` ： headless（），。`--gui`  CartPole 。 GUI （ 16ms），； GUI ，。

### **： SwanLab ？**

 SwanLab  `mode="local"`，****。，：

```bash
swanlab watch swanlog
```

：

- `http://127.0.0.1:5092`
- `http://localhost:5092`

， SwanLab ：

![ SwanLab ](./images/cartpole-swanlab-dashboard.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em> <code>http://127.0.0.1:5092</code> ，。；， <code>Chart</code> ，。</em>
</div>

，：

![ SwanLab ](./images/cartpole-swanlab-experiment-chart.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em>、 <code>Chart</code> 。 <code>rollout</code>、<code>time</code>、<code>train</code> 。</em>
</div>

 `rollout/ep_rew_mean`，””。，。

 SwanLab ，：

- `https://swanlab.cn`

，。，。

， [](./metrics)。

```python
#  SB3 ； PyTorch ， PPO 
import gymnasium as gym
from stable_baselines3 import PPO
from swanlab.integration.sb3 import SwanLabCallback

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=1)

# （SwanLab ）
model.learn(
    total_timesteps=80000,
    callback=SwanLabCallback(
        project="cartpole-ppo",
        experiment_name="PPO-CartPole-v1",
        mode="local",
    ),
)

# 
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"！: {mean_reward} +/- {std_reward}")
model.save("output/ppo_cartpole")

# （ --gui  render_mode="human" ，）
vis_env = gym.make("CartPole-v1", render_mode="human")  #  None
for episode in range(5):
    obs, info = vis_env.reset()
    ...
```

，。，？「」「」。
