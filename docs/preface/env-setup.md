# 

> ****：， Python、PyTorch、RL  LLM 。，。

## （5 ）

， 4  6 。 LLM 。

::: code-group

```bash [conda（）]
conda create -n rl-course python=3.10 -y
conda activate rl-course
pip install torch torchvision
pip install gymnasium stable-baselines3[extra]
pip install numpy scipy matplotlib tqdm
```

```bash [venv]
python3.10 -m venv rl-course
source rl-course/bin/activate   # Windows: rl-course\Scripts\activate
pip install torch torchvision
pip install gymnasium stable-baselines3[extra]
pip install numpy scipy matplotlib tqdm
```

:::

—— `CartPole` ：

```python
import gymnasium as gym
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA:    {torch.cuda.is_available()}")

env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()
for _ in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
print("！")
```

::: details  GPU 
（CartPole、DQN ）CPU ， GPU。 LLM （Ch7-Ch10） 24 GB ， [Google Colab](https://colab.research.google.com/)  GPU 。
:::

---

**，。**

## Python 

 **Python 3.10+**（3.10、3.11  3.12 ）。 conda ， CUDA 。

```bash
# ： conda（）
conda create -n rl-course python=3.10 -y
conda activate rl-course

# ： venv（）
python3.10 -m venv rl-course
source rl-course/bin/activate  # Linux/macOS
# rl-course\Scripts\activate   # Windows
```

::: tip  Python 3.10？
PyTorch 2.x、transformers 4.x  gymnasium  3.10 。3.12 ，。
:::

## PyTorch 

PyTorch 。。

**： GPU **

```bash
#  NVIDIA  CUDA 
nvidia-smi
```

 `CUDA Version: 12.x`， CUDA 。

**： PyTorch**

```bash
# CUDA 12.4（，）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CUDA 11.8（）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU （ GPU ）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Apple Silicon（M1/M2/M3/M4 Mac）
pip install torch torchvision
```

**： PyTorch GPU**

```python
import torch
print(f"PyTorch : {torch.__version__}")
print(f"CUDA : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"GPU : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

##  RL 

（Ch1-Ch6）。

```bash
# Gymnasium（Gym ，）
pip install gymnasium

# Stable-Baselines3（ RL ）
pip install stable-baselines3[extra]

# 
pip install numpy scipy matplotlib seaborn pandas

# 
pip install tqdm tensorboard wandb
```

## 

，。

```bash
# 11：PyBullet 
pip install pybullet

# 4：Atari （ ale-py）
pip install "gymnasium[atari,accept-rom-license]"
pip install ale-py

# 4：ViZDoom 3D
pip install vizdoom

# 4：stable-retro（）
pip install stable-retro
```

**MuJoCo （，）**

```bash
# MuJoCo ， pip 
pip install mujoco
#  gymnasium  MuJoCo 
pip install "gymnasium[mujoco]"
```

::: warning MuJoCo 
MuJoCo  GPU 。， `export MUJOCO_GL=egl`  `export MUJOCO_GL=osmesa`。
:::

**Isaac Lab （， GPU ）**

Isaac Lab  NVIDIA Isaac Gym ， GPU ， RL 。 NVIDIA GPU + Linux。

```bash
# Isaac Lab  Isaac Sim， Isaac Sim
pip install isaacsim[all]

#  Isaac Lab 
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install

# 
python -c "import isaacsim; print('Isaac Lab ready')"
```

::: warning Isaac Lab 
Isaac Lab  Linux （Ubuntu 22.04 ） NVIDIA RTX  GPU（ 8GB ）。macOS 。 macOS， Isaac Lab， PyBullet  MuJoCo 。
:::

**Unity ML-Agents （， 3D  RL）**

Unity ML-Agents  Unity  3D  RL ，、。

```bash
#  Python  mlagents 
pip install mlagents-envs

# （ PyTorch ）
pip install mlagents

# 
python -c "from mlagents_envs.environment import UnityEnvironment; print('ML-Agents ready')"
```

 ML-Agents  Unity （`.exe` / `.app` / Linux ）。 [ML-Agents GitHub Releases](https://github.com/Unity-Technologies/ml-agents/releases) 。[](../appendix_game_projects/intro)。

::: tip Unity ML-Agents 
ML-Agents **3D **：Atari  2D ，CartPole ， ML-Agents  3D （、、）。、 3D ，ML-Agents  Gymnasium/PyBullet 。
:::

## LLM 

（Ch7-Ch10）。

```bash
# Hugging Face 
pip install transformers datasets accelerate peft

# TRL（Transformer Reinforcement Learning，DPO/PPO ）
pip install trl

# （，）
pip install bitsandbytes

# 
pip install lm-eval
```

::: tip 
，：

```bash
pip install "transformers==4.57.3" "trl==0.24.0" "datasets==4.4.1" "accelerate==1.10.1" "peft==0.17.1"
```

:::

## GPU 

，。

|       |                              |                 |
| ----------- | -------------------------------- | ----------------------- |
| NVIDIA  | `nvidia-smi`                     |  GPU  |
| CUDA    | `nvcc --version`                 |  PyTorch  |
| cuDNN       | `torch.backends.cudnn.version()` |               |
| PyTorch GPU | `torch.cuda.is_available()`      | `True`                  |

```python
# 
import torch
print("=" * 50)
print("")
print("=" * 50)
print(f"Python:       {__import__('sys').version}")
print(f"PyTorch:      {torch.__version__}")
print(f"CUDA :    {torch.cuda.is_available()}")
print(f"CUDA :    {torch.version.cuda or 'CPU'}")
if torch.cuda.is_available():
    print(f"cuDNN:        {torch.backends.cudnn.version()}")
    print(f"GPU:          {torch.cuda.get_device_name(0)}")
    print(f":         {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Apple MPS:    {torch.backends.mps.is_available()}")
print("=" * 50)
```

## 

###  1：CUDA 

****：`RuntimeError: CUDA out of memory`  `Found no NVIDIA driver`

****：PyTorch  CUDA 。

```bash
#  PyTorch  CUDA 
python -c "import torch; print(torch.version.cuda)"

# ， PyTorch
# 
pip uninstall torch torchvision -y
# （ PyTorch ）
```

###  2：

****：`ImportError`  `AttributeError`，。

```bash
# 
pip list | grep -E "torch|gymnasium|transformers|trl"

# 
pip install --force-reinstall <>
```

###  3：Apple Silicon (M ) 

```bash
# M  MPS 
#  arm64  PyTorch
python -c "import platform; print(platform.processor())"
#  arm64

#  RL  Mac ， headless 
# export PYOPENGL_PLATFORM=egl
```

###  4：MuJoCo 

```bash
# 
export MUJOCO_GL=egl
#  osmesa
export MUJOCO_GL=osmesa
```

## 

， "OK"，。

```python
"""

: python verify_env.py
"""
import sys

checks = []

# 1. Python 
ver = sys.version_info
checks.append(("Python >= 3.10", ver >= (3, 10)))

# 2. 
for pkg in ["numpy", "matplotlib", "scipy", "tqdm"]:
    try:
        __import__(pkg)
        checks.append((pkg, True))
    except ImportError:
        checks.append((pkg, False))

# 3. PyTorch
try:
    import torch
    checks.append(("PyTorch", True))
    checks.append(("CUDA GPU", torch.cuda.is_available()))
except ImportError:
    checks.append(("PyTorch", False))

# 4. RL 
try:
    import gymnasium
    checks.append(("Gymnasium", True))
except ImportError:
    checks.append(("Gymnasium", False))

# 5. LLM 
for pkg in ["transformers", "datasets", "peft", "accelerate", "trl"]:
    try:
        __import__(pkg)
        checks.append((pkg, True))
    except ImportError:
        checks.append((pkg, False))

# 
print("\n" + "=" * 45)
print("  ")
print("=" * 45)
for name, ok in checks:
    status = "OK" if ok else "MISSING"
    print(f"  {name:<25} {status}")
print("=" * 45)
passed = sum(1 for _, ok in checks if ok)
print(f"  : {passed}/{len(checks)}")
print("=" * 45 + "\n")
```

## 

|               |  |                             |      |
| ----------------- | -------- | ------------------------------- | ------------ |
| Python            | 3.10+    |                           |          |
| PyTorch           | 2.1+     |                     |          |
| gymnasium         | 0.29+    | RL                      | Ch1, Ch3-Ch6 |
| stable-baselines3 | 2.2+     |  RL                 | Ch1, Ch4-Ch6 |
| numpy             | 1.24+    |                         |          |
| matplotlib        | 3.7+     |                       |          |
| pybullet          | 3.2+     |                       | Ch11         |
| mujoco            | 3.0+     |                   | Ch11         |
| isaacsim          | 4.0+     | GPU （Isaac Lab） | Ch11, Ch12   |
| mlagents          | 1.0+     | Unity 3D  RL            |          |
| ale-py            | 0.8+     | Atari                     | Ch4          |
| transformers      | 4.45+    | LLM                     | Ch7-Ch10     |
| trl               | 0.12+    | LLM                 | Ch7-Ch8      |
| peft              | 0.13+    |  (LoRA)             | Ch7-Ch10     |
| accelerate        | 1.0+     |                       | Ch7-Ch10     |
| datasets          | 3.0+     |                       | Ch7-Ch10     |
| wandb             | 0.16+    |                         |          |
| tensorboard       | 2.15+    |                       |          |

::: tip 
：（Python 、PyTorch、 RL ）， LLM ，。
:::
