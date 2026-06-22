---
title: PPO 
description:  PPO ，，， PPO 、、。
outline:
  level: [2, 3]
---

# 7.5 PPO 

 PPO ，。 PPO ，""""。

## 

，：

1. ** PPO **：、、。
2. **""""**：PPO ；、。
3. ****： Flappy Bird， Pokemon Red，。

## 

""，****：

|          |              |        |             |
| ------------ | ---------------- | ------------ | ----------------- |
| **** |      |      |  +  |
| **** |      | 、   | 、        |
| **** |  |  |   |

Flappy Bird ，；Pokemon Red ，。， PPO 、、。

---

### Flappy Bird：

![Flappy Bird PPO ，：wangjia184/rl ](https://user-images.githubusercontent.com/44725090/67148880-e7dba280-f2a4-11e9-8dbf-d154842ee0cf.gif)

Flappy Bird "**PPO **"。——、、、，：。****，。

**Dhyanesh18**  **Stable-Baselines3**， `CnnPolicy` ， **4 **，**** callback。，****： 2-3  MLP ， **$3 \times 10^{-4}$**，gamma **0.99**，PPO clip **0.2**， episode 。， **10M timesteps** 。

——"** →  →  → **"， PPO  bug。

|  |                                                                                                                                                                         |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  | [Flappy Bird ](https://flappybird.gg/)                                                                                                                              |
|  | [flappy-bird-gymnasium](https://pypi.org/project/flappy-bird-gymnasium/)                                                                                                    |
| PPO  | [Yuanpeng-Li/Flappy-Bird-AI](https://github.com/Yuanpeng-Li/Flappy-Bird-AI)（PPO ）、[Dhyanesh18/flappbird-rl](https://github.com/Dhyanesh18/flappbird-rl)（PPO + A2C） |

---

### Snake：

![Snake PPO2 ，：gym-snake-rl ](https://d1k6fapei95iy6.cloudfront.net/imgs_taemin/ppo2-fullobs.gif)

Snake  Flappy Bird ：****。，，。

****。"** +10， -10**"，****——，。********，。 **Stable-Baselines PPO2** ，5×5  **23**（），10×10  **100M steps**  **6-7 **。 **DQN** ，PPO ， **900k steps**  DQN  **140k**，****。

"****"——，。 PPO ，，。

|  |                                                                                                                      |
| -------- | ------------------------------------------------------------------------------------------------------------------------ |
|  | [Snake ](https://snaketap.com/)                                                                                  |
|  | [gym-snake-rl](https://jfpettit.svbtle.com/introducing-gym-snake-rl)、[Gym-Snake](https://github.com/grantsrb/Gym-Snake) |
| PPO  | [Introducing gym-snake-rl](https://jfpettit.svbtle.com/introducing-gym-snake-rl)（PPO2 ）                        |

---

### 2048：

![2048 PPO ，：tejpshah/2048-DeepRL](https://github.com/tejpshah/2048-DeepRL/raw/main/gifs/PPO.gif)

2048 ，：****。， tile，。"****""****"，。

**arturf1**  **Unity ML-Agents PPO** 。****（ +  tile  +  2048  bonus） **215k ** **4%**；****（ tile ） **450k ** **37%**。——****。**tejpshah**  PyTorch PPO  one-hot  4×4 ， **512 tile**， **256**； **DDQN**  **1024**  **2048**。

，：** PPO ，**。，****—— 2048 ，。

|  |                                                                                                                                                        |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  | [2048 ](https://classic.play2048.co/)                                                                                                          |
|  | [gymnasium-2048](https://pypi.org/project/gymnasium-2048/)、[2048 ](https://github.com/gabrielecirulli/2048)                                           |
| PPO  | [arturf1/2048](https://github.com/arturf1/2048)（Unity ML-Agents PPO）、[tejpshah/2048-DeepRL](https://github.com/tejpshah/2048-DeepRL)（PPO + DDQN ） |

---

### Tetris：

![Tetris ，：Wikimedia Commons](https://upload.wikimedia.org/wikipedia/commons/5/5c/Tetris_freemade.png)

Tetris 。，。，。

 bitboard  PPO （arXiv:2603.26765） **afterstate-evaluating actor **， 10×10  **61,440 steps**（ 3 ） **3829**， **4124**。 10×20 ，——** Tetris ，**。，**PPO  Tetris **，"**RL **"。

 + ， RGB 。、、、。

|  |                                                                                                                       |
| -------- | ------------------------------------------------------------------------------------------------------------------------- |
|  | [Tetris ](https://play.tetris.com/)                                                                           |
|  | [ALE Tetris](https://ale.farama.org/environments/tetris/)、[tetris-gymnasium](https://pypi.org/project/tetris-gymnasium/) |
| PPO  | [chirbard/ppo-Tetris-v5](https://huggingface.co/chirbard/ppo-Tetris-v5)（PPO ）                                       |

---

### Breakout：

![Breakout PPO ，：Stable-Baselines3 ](https://stable-baselines3.readthedocs.io/en/master/_images/breakout.gif)

Breakout "****"。、、，。**Stable-Baselines3  PPO**  baseline：**10M steps**  **398 ± 33**，**5M steps**  **300 **（）。 **360-430** 。

****。，****（ **4 **）。********——，100k steps  **150 **， RGB  **-48 **。，，。**ICLR Blog Track 2022**  PPO ，Stable-Baselines3  **398 ** CleanRL、Tianshou ， Atari ****。

|  |                                                                                                                                                                          |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  | [Atari Breakout ](https://brickbreaker.app/atari-breakout/)                                                                                                          |
|  | [ALE Breakout](https://ale.farama.org/environments/breakout/)                                                                                                                |
| PPO  | [sb3/ppo-BreakoutNoFrameskip-v4](https://huggingface.co/sb3/ppo-BreakoutNoFrameskip-v4)（Stable-Baselines3 PPO）、[CleanRL PPO](https://docs.cleanrl.dev/rl-algorithms/ppo/) |

---

### Procgen： vs 

![Procgen ，：OpenAI procgen GitHub](https://raw.githubusercontent.com/openai/procgen/master/screenshots/procgen.gif)

Procgen ""，****。，PPO ，？OpenAI  benchmark  **IMPALA-CNN** ， **$5 \times 10^{-4}$**，gamma **0.999**，****。

 `coinrun` ——，，PPO  easy ** 10**， **8.31 ± 0.12**。，（ **IDAAC**、**DAAC**） CoinRun 。：****，"**、、**"。

|  |                                                                                                                                                                          |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  | [Procgen PyPI](https://pypi.org/project/procgen/)， `python -m procgen.interactive --env-name coinrun`                                                                   |
|  | [OpenAI Procgen Benchmark](https://openai.com/index/procgen-benchmark/)、[procgen PyPI](https://pypi.org/project/procgen/)                                                   |
| PPO  | [OpenAI Procgen Benchmark](https://openai.com/index/procgen-benchmark/)（ PPO ）、[CleanRL ppo_procgen.py](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppoprocgenpy) |

---

### CarRacing： + 

![CarRacing PPO ，：Solving CarRacing with PPO](https://notanymike.github.io/img/posts/SolvingCarRacing/4.gif)

CarRacing 。 **96×96** ，****：、、。——****，100k steps  **150 ** RGB  **-48**。** 4 **，。

：**400k-500k steps**  **450-620 **，；**2M steps**  **740-920+ **，；， **917 **。 **A2C** ——A2C ****， **-90**， **PPO  clipped surrogate objective **。

 BipedalWalker ，，。 **LSTM** ，，。

|  |                                                                                                                                                                                  |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
|  | [Gymnasium CarRacing ](https://gymnasium.farama.org/environments/box2d/car_racing/)                                                                                              |
|  | [Gymnasium CarRacing-v3](https://gymnasium.farama.org/environments/box2d/car_racing/)                                                                                                |
| PPO  | [Solving CarRacing with PPO](https://notanymike.github.io/Solving-CarRacing/)（PPO ）、[Rinnnt/ppo-CarRacing-v3](https://huggingface.co/Rinnnt/ppo-CarRacing-v3)（PPO ） |

---

### Super Mario Bros：

![Super Mario Bros PPO ，：vietnh1009/Super-mario-bros-PPO-pytorch](https://github.com/vietnh1009/Super-mario-bros-PPO-pytorch/raw/master/demo/video-1-1.gif)

Mario ，****。，****；，。**vietnh1009**  PyTorch  **CNN Actor-Critic**，、、resize  **84×84**、**4 **。 **$10^{-3}$**  **$10^{-5}$** ， World 1-3 （ **$7 \times 10^{-5}$**）。

 **31/32 **—— **A3C**  **19/32** 。**PPO  clipped surrogate objective** ：**A3C  19 ， PPO  12 **。

 NES 。 **`RIGHT_ONLY`**（**2 **） **`SIMPLE_MOVEMENT`**（**5 **）， World 1-1 。：****；****；****。

|  |                                                                                                                                                                                    |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  | [gym-super-mario-bros CLI](https://github.com/Kautenja/gym-super-mario-bros)： `gym_super_mario_bros -e SuperMarioBros-v0 -m human`                                                |
|  | [Kautenja/gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)、[PyPI](https://pypi.org/project/gym-super-mario-bros/)                                              |
| PPO  | [vietnh1009/Super-mario-bros-PPO-pytorch](https://github.com/vietnh1009/Super-mario-bros-PPO-pytorch)（PPO ）、[super-mario-agent](https://github.com/nemanja-m/super-mario-agent) |

---

### Sonic / Gym Retro：

![Sonic / Retro Contest ，：OpenAI Retro Contest Results](https://images.ctfassets.net/kftzwdyauwt9/1gGJCzTHFTNJn6m4GkROAB/6cb51a280c362510874d3ee15f5da6e8/retro-contest-results.jpg?w=3840&q=90&fm=webp)

**Retro Contest 2018** ****：，。 **7438 **， **10000**。OpenAI  **Joint PPO baseline**  **3128 **——****。

**（Dharmaraja）**：****。****， PPO **fine-tune** ， **4692**。 **Rainbow DQN** ，。： **PPO  fine-tune **；，** 60% **。

 PPO ：****，。 **PPO **——，****。

|  |                                                                                                                                                                         |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  | [Gym Retro interactive script](https://retro.readthedocs.io/en/latest/getting_started.html)（ ROM）                                                               |
|  | [Gym Retro ](https://retro.readthedocs.io/en/latest/)、[OpenAI Gym Retro](https://openai.com/research/gym-retro)                                                        |
| PPO  | [OpenAI Retro Contest](https://openai.com/index/retro-contest/)（PPO baseline）、[Retro Contest Results](https://openai.com/index/retro-contest-results/)（Joint PPO ） |

---

### Unity SoccerTwos：

![Unity SoccerTwos PPO ，：Hugging Face Deep RL Course](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit10/soccertwos.gif)

SoccerTwos  2v2 ，：****。PPO  **self-play** ——， **ELO **。**Unity ML-Agents  PPO** ，batch_size **2048**、buffer_size **20480**、 **$3 \times 10^{-4}$**、 **0.005**。

****。，；ELO  **1200**  **1600** ，**-**。**SAC** ， **1200-1250**， **PPO  competitive **。

****。（、） PPO  **250k iterations**  **1.8**， **40%** ；（**0.35 vs 0.45**），****。

****。，；，。****，。

|  |                                                                                                                                                                             |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  | [Hugging Face Space: ML-Agents SoccerTwos](https://huggingface.co/spaces/unity/ML-Agents-SoccerTwos)                                                                            |
|  | [Unity ML-Agents ](https://unity-technologies.github.io/ml-agents/Training-ML-Agents/)、[HF Deep RL Course](https://huggingface.co/learn/deep-rl-course/unit7/introduction) |
| PPO  | [blu666/ppo-SoccerTwos](https://huggingface.co/blu666/ppo-SoccerTwos)（PPO ）、[Sekiraw/SoccerTwos](https://huggingface.co/Sekiraw/SoccerTwos)（PPO ）                  |

---

### ViZDoom：

![ViZDoom PPO ，：GuillBla/RL-Doom](https://github.com/GuillBla/RL-Doom/raw/master/demos/demo_basic.gif)

****——，，。**GuillBla**  **Stable-Baselines3**  `CnnPolicy`，****、resize  **100×160**、 UI  **85×160**。

。**Basic** ：3 （、、）， **100k steps** 。**Defend** ：3 （、、）， **200k steps** ，。**Deadly Corridor** ：7 （ +  + ），********，200k steps ，。

**Deadly Corridor **—— PPO ** 3D  + **，****。/ **LSTM** ，****，****。

|  |                                                                                                                                                                      |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
|  | [ViZDoom ](https://vizdoom.farama.org/environments/default/)、[Freedoom](https://github.com/freedoom/freedoom)                                                   |
|  | [ViZDoom ](https://vizdoom.farama.org/)、[ViZDoom default scenarios](https://vizdoom.farama.org/environments/default/)                                               |
| PPO  | [GuillBla/RL-Doom](https://github.com/GuillBla/RL-Doom)（Stable-Baselines3 PPO）、[callumhay/vizdoom_ppo_rnd](https://github.com/callumhay/vizdoom_ppo_rnd)（PPO + RND） |

---

### Pokemon Red：

![Pokemon Red ，：PWhiddy/PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments/raw/master/assets/grid.png?raw=true)

Pokemon Red **、 flag、、、**——****。**PWhiddy**  baseline  **PPO + PyBoy** ，V2 **** KNN ，， **Cerulean City**（）。

 PPO ****。****——，****。** spam**——，。**PokeRL**（arXiv:2604.10812） PPO  **Loop-Aware Environment Wrapper**  **Anti-Spam Mechanism**，"**、、**"，。

****：****，**PPO **。** wrapper** ，****。

|  |                                                                                                                                                          |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
|  | [PokemonRedExperiments ](https://github.com/PWhiddy/PokemonRedExperiments)（）                                                     |
|  | [PWhiddy/PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments)、[davidpaulius/PokemonRedRL](https://github.com/davidpaulius/PokemonRedRL) |
| PPO  | [PokeRL ](https://drubinstein.github.io/pokerl/)（PPO ）、[PokeRL ](https://arxiv.org/abs/2604.10812)（PPO baseline）                          |

---

### Crafter：PPO 

![Crafter ，：danijar/crafter GitHub](https://raw.githubusercontent.com/danijar/crafter/main/media/video.gif)

Crafter 、、、、 **22 **， Minecraft 。， **22 **—— favor ****。 **50.5%**， PPO（**Stable-Baselines3 CNN**） **4.6%**。 **Impala ResNet**  **15.6%**， DreamerV3 。

。**、**；**、**；**、** **unreachable**。**EnvGen** ，PPO  **1.96M steps** "" **135k steps** （ 1M window），"" **925k steps**。 LLM ** fine-tune**， **40k**  **192k steps** ——** PPO ，**。

：**PPO ""，"" 10+ **。

|  |                                                                                                                |
| -------- | ------------------------------------------------------------------------------------------------------------------ |
|  | [danijar/crafter](https://github.com/danijar/crafter)： `python3 -m crafter.run_gui`                           |
|  | [Crafter GitHub](https://github.com/danijar/crafter)、[Crafter project page](https://danijar.com/project/crafter/) |
| PPO  | [Benchmarking and Improving RL Generalization with Crafter](https://arxiv.org/abs/2208.03374)（PPO baseline ） |

---

### MiniHack / NetHack：

![MiniHack / NetHack ，：hr0nix/omega](https://github.com/hr0nix/omega/raw/main/images/river.gif)

MiniHack  NetHack 、、、、。，** ASCII/Unicode **——， **CNN  Transformer** 。

 **NetHack Learning Environment（NLE）** ，**Sample Factory**  PPO  **480 ** RTX 2080Ti  **24 ** **700 ** NetHackScore， TorchBeast baseline（ **400 **）。 MiniHack ，PPO ****——ZombieHorde  **5 **，TreasureDash  **20-23 **（ **28**）。， **RND**、**NovelD**、**E3B** **** PPO baseline，**""**， **PPO **。

（**SOL**、**MOTIF**）：** PPO **， PPO ****。PPO "** lower bound**"——。

|  |                                                                                                                                                                  |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  | [MiniHack trying out](https://minihack.readthedocs.io/en/latest/getting-started/trying_out.html)： `python -m minihack.scripts.play_gui --env MiniHack-River-v0` |
|  | [MiniHack ](https://minihack.readthedocs.io/)、[MiniHack GitHub](https://github.com/facebookresearch/minihack)                                                   |
| PPO  | [hr0nix/omega: PPO and MuZero agents](https://github.com/hr0nix/omega)（PPO + MuZero）                                                                               |

---

## 

|                |                | PPO                              |
| ------------------ | ------------------------ | ---------------------------------------- |
| Flappy Bird        |  PPO             | ，               |
| Snake              |      | 5×5  23 ，10×10  6-7         |
| 2048               |  vs      |  37% ， DQN            |
| Tetris             | ，RL vs  | 10×10  3800 ，         |
| Breakout           |      | 10M steps  398 ，            |
| Procgen            |  vs              | CoinRun  8.31±0.12，   |
| CarRacing          |          | 2M steps 740-920+ ，A2C  -90         |
| Super Mario Bros   |      | 31/32 ， A3C                 |
| Sonic / Gym Retro  |            |  4692 ， 7438            |
| Unity SoccerTwos   |            | ELO ~1600，            |
| ViZDoom            |              | Basic/Defend ，Corridor        |
| Pokemon Red        |        |  Cerulean City， anti-loop wrapper |
| Crafter            |                  | CNN  4.6%，ResNet  15.6%             |
| MiniHack / NetHack |        | NetHackScore ~700，          |

---

## 

****： Flappy Bird、Snake、2048、Breakout、CarRacing。，，。

**""**： Procgen、Mario、SoccerTwos、ViZDoom、Crafter。，。

****： Pokemon Red 、Super Mario Bros World 1-1、Sonic / Gym Retro 。，。

****： Crafter、MiniHack、Pokemon Red。 PPO ：、、、。

---

## 

1.  Flappy Bird ，""？？
2.  Snake ，" +10， -10"，？？
3. Breakout ， 2048 。？
4. Procgen  Retro Contest ，。Procgen  procedurally generated ，Retro 。？
5. Pokemon Red ， PPO（）？，？

---

## 

PPO ，""""。， PPO ：

- **、**，PPO 、。
- ****，。
- ****，。
- **、、** PPO ，，。

，""，" PPO "。
