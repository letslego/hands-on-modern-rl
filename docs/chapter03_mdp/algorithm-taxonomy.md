# 3.7 ：On-policy、Off-policy、Online  Offline

> ****：——**On-policy vs Off-policy**（？）， **Online vs Offline**（？）。，。

## 

****

-  $\mu$  $\pi$：“”“”。
- On-policy / Off-policy：。
- Online / Offline：、。

，， TD 。：****。 $V(s)$  $Q(s,a)$，、、（ $s, a, r, s'$）。

，：**？**

：

- ****：，，。
- ****：，（），。
- ****：《》，，。

。，，****。

，。

::: info 

- **On-policy（）**：（）。
- **Off-policy（）**：、，（）。
- **Online（）**：，（）。
- **Offline（）**：，（）。
  :::

## ：，？

，“”。

， $\pi$：，。 RL，。

：

1. **（Behavior Policy）**， $\mu(a\mid s)$。
   ： **“？”**
   （）、、 token 。 $\epsilon$-greedy ，，，。

2. **（Target Policy）**， $\pi_\theta(a\mid s)$。
   ： **“？”**
   。， $\theta$ ，。

“（）”“（）”，。

## ：On-policy  Off-policy

：**，？**

### On-policy（）：

（）， **On-policy（）**。

，：

$$
\mu(a\mid s) \approx \pi_\theta(a\mid s)
$$

****：，，，****。。

， **Sarsa**  On-policy 。 $s$  $a$， $s'$，**** $s'$  $a'$ 。“，”。

（LLM），**PPO**（ DeepSeek  **GRPO**） On-policy 。 prompt （Rollout），，。 On-policy ，PPO （Clipping ），。[^ppo2017]

- ****：，。，。
- ****：（Sample Inefficient）。，，“”，。

### Off-policy（）：

 $\mu$  $\pi_\theta$ ， **Off-policy（）**。

$$
\mu(a\mid s) \neq \pi_\theta(a\mid s)
$$

****：（Replay Buffer），、，。（ $\pi$），（ $\mu$），。

 Off-policy  **Q-Learning**。：

$$
\text{Target} = r + \gamma \max_{a'} Q(s', a')
$$

，（ $\mu$ ）， Q ，****（ $\pi$ ）。，。[^watkins1992]

，**DQN**  Off-policy 。（Replay Buffer），。， $\mu \neq \pi$， Off-policy 。[^mnih2015]

- ****：（Sample Efficient）。，。
- ****：（Distribution Shift）。 Off-policy 。，，“”（），。[^sutton-barto]

## ：Online  Offline

On-policy  Off-policy “”。，：**，，？**

 Online  Offline 。****。

### Online RL（）：

，，， **Online RL（）**。

，：

$$
\mathcal{D}_{k+1} = \mathcal{D}_k \cup \{\tau_k\}
$$

 DQN  Atari ， PPO ，，、、。

，**Online  On-policy**。DQN  Off-policy ，； Online ，。

Online RL ****。，，，。

### Offline RL（）：

，，****， **Offline RL（）**。

$$
\mathcal{D} = \mathcal{D}_{\text{fixed}}
$$

？，“”。[^levine2020]

- ****： AI “”。（），。
- ****： AI 。

（LLM），**DPO（）**  Offline RL。（Prompt +  + ），。，。[^dpo2023]

**Offline RL ：**
， Off-policy （）。Offline **（Overestimation）**。

。AI ，，：“ 120km/h ，！”
 Online RL，AI ，，。 Offline RL ，，，。

，Offline RL （ CQL）**（Conservatism）**：，， AI 。[^cql2020]

## ：

，。，：

|                  |                                                        |                                    |                                        |
| ------------------------ | ---------------------------------------------------------- | ------------------------------------------------ | ---------------------------------------------- |
| **Online + On-policy**   | ，。           | REINFORCE、Sarsa、PPO、GRPO                      | ，。             |
| **Online + Off-policy**  | ，。       | Q-Learning、DQN、SAC、TD3                        | 。               |
| **Offline + Off-policy** | ，。 | CQL、IQL、 DPO                     | （）。 |
| **Offline + On-policy**  | ，。                     | （）、 | ，。             |

## 

，：

1. **“On-policy ？”**
   。PPO  On-policy， epoch。：（ Clipping ）， On-policy 。

2. **“Off-policy ？”**
   。Off-policy ，**（Coverage）**。，，。。

3. **“Off-policy  Offline？”**
   。**DQN** 。（Off-policy），（Online）。

4. **“DPO ， RL？”**
    DPO ，，。 Offline ，， Offline RL 。

## ：On-policy ？

，On-policy  $\mu$  $\pi_\theta$ 。，——？：**（Training-Inference Mismatch）**。

，—— RL ，，。AlphaGo、Atari DQN （Policy Lag）。** RL **， LLM-RL ，，。

> **"When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch"**
> _(Richard Li et al., 2025)_

： LLM-RL ，$\pi_{\text{rollout}}$（） $\pi_{\text{old}}$（""）****。

- ****（ rollout ）：vLLM / SGLang，FP8/BF16 ，KV Cache 
- ****（ log prob ）：FSDP/Megatron，BF16/FP32 ，

、， log-probability ****。 $\mu$  $\pi_\theta$， $\mu \approx \pi_\theta$ ""。

> **"Defeating the Training-Inference Mismatch via FP16"**
> _(Qi et al., 2025)_

。BF16 ， token  log-probability 。 FP16，—— LLM-RL 。

> **"Taming the Tail: Stable LLM Reinforcement Learning via Dynamic Vocabulary Pruning"**
> _(arXiv 2512.23087, 2025)_

****： $(1-p)$ —— token ， token ，，。

> **"Stabilizing Reinforcement Learning with LLMs: Formulation and Practices"**
> _(Zheng et al., Qwen Team, arXiv 2512.01374, 2025)_

 Qwen ：token-level  REINFORCE ****，——**(1) **，**(2) **。，。

###  PPO 

： PPO ？：**PPO  Clipping ""，**。

PPO ：

$$
\mathcal{L}^{\text{CLIP}} = \mathbb{E}\left[\min\left( r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right)\right]
$$

 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$ 。PPO  Clipping  $r_t$  1 ，："，，。"

 PPO  Clipping ——** $\pi_{\text{old}}$ ""**。

 RL（Atari、MuJoCo ）， Python ，$\pi_{\text{old}}$ ，。 PPO  Clipping ""，。

 LLM-RL ，：

- $\pi_{\text{rollout}}$：vLLM  FP8 ****
- $\pi_{\text{old}}$： BF16/FP32 ""

****。， $r_t$ ****——PPO  Clipping ，。

：PPO  Clipping ****，""。****，Clipping 。

 LLM-RL  PPO Clipping，。，：

- ****：FP16/BF16  FP8  Rollout， $\pi_{\text{rollout}}$  $\pi_{\text{old}}$ （Qi et al., 2025）；——FP8-RL  veRL  W8A8 ，，Rollout  44%  BF16 （Qiu et al., arXiv 2601.18150）。
- **（IS）**： $\pi_{\text{rollout}} \neq \pi_{\text{old}}$，。Truncated IS（TIS）， IS （Yao et al., NeurIPS 2025）； MinPRO（Lei et al., arXiv 2601.22718）， token ， Off-policy 。
- ** token**：， token （"Taming the Tail", arXiv 2512.23087）。
- **MoE **： Expert ，R3（Rollout Routing Replay）， MoE-RL （Zheng et al., arXiv 2512.01374）。
- ****：，（Zhang et al., arXiv 2602.01826）。
- ****： Rollout  log-probability， $\pi_{\text{rollout}}$  $\pi_{\text{old}}$——。

### 

： LLM-RL ，"" On-policy。** $\mu$  $\pi_\theta$ **——PPO  Clipping ，FP16 ，R3 。 On/Off-policy ，****—— On-policy， Off-policy 。

## 

，“”：

1. ** $\mu$** ，** $\pi_\theta$** 。
2. **On-policy vs Off-policy** ：？
3. **Online vs Offline** ：？

，（Bellman）、（DP/MC/TD）、（On/Off-policy）。，： **** 。

、： **，？**

：[](./reward-design)

## ：

RL  Agentic RL 。，——。 2024–2026 ，。

### On-policy  Off-policy：？

 RL 。****。

**：**

> **"Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm: Demystifying Some Myths About GRPO and Its Friends"**
> _(arXiv 2509.24203, 2025)_

——DeepSeek  GRPO  On-policy ， Off-policy。 **"Secretly an Off-Policy Algorithm"** ：（On-policy），（Off-policy）。

> **"Prosperity before Collapse: How Far Can Off-Policy RL Reach with Stale Data on LLMs?"**
> _(arXiv 2510.01161, 2025)_

 **"Off-Policy RL"** + **"Stale Data"**（） Off-policy ：（stale），。 M2PO ，， Off-policy  1.7B–32B  On-policy 。

> **"On-Policy RL Meets Off-Policy Experts: Harmonizing Supervised Fine-Tuning and Reinforcement Learning via Dynamic Weighting"**
> _(arXiv 2508.11408, 2025)_

 **On-Policy**  **Off-Policy** 。"On-Policy RL" 、 RL ；"Off-Policy Experts"  SFT （""，）。 CHORD —— LLM  On/Off-policy 。

> **"Behaviour Policy Optimization: Provably Lower Variance Return Estimates for Off-Policy Reinforcement Learning"**
> _(arXiv 2511.10843, AAAI 2026)_

 **"Behaviour Policy"**（） **"Off-Policy"** ，：Off-policy ， $\mu$  $\pi$ ， On-policy 。

**：**  **On-policy**，""； **Off-policy**，"（）"。

### Online  Offline：？

****， On/Off-policy 。

**：**

> **"Offline vs. Online Learning in Model-based RL: Lessons for Data Collection Strategies"**
> _(arXiv 2509.05735, RLC 2025)_

 **Offline**  **Online** 。 31 ，，（OOD）—— Offline RL ：，。

> **"Understanding the Performance Gap Between Online and Offline Alignment Algorithms"**
> _(arXiv 2405.08448, NeurIPS 2024)_

 **"Online and Offline Alignment"**  LLM 。Online alignment  PPO ，Offline alignment  DPO 。 Online  Offline 。

**：**  **Online**，、； **Offline**，，。

### ：

 On/Off-policy  Online/Offline ，：

|                      |                                                                                                                           |                                                                                                                                                  |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Online + On-policy**   | PPO (Schulman et al., 2017)、GRPO (DeepSeek, 2024)                                                                                | ，。                                                                                                                                         |
| **Online + Off-policy**  | _"TOP-ERL: Transformer-based Off-Policy Episodic Reinforcement Learning"_ (ICLR 2025 Spotlight)                                   | **Off-Policy** ， **Episodic** 、（Online）。                                                      |
| **Offline + Off-policy** | _"Offline-Boosted Actor-Critic: Adaptively Blending Optimal Historical Behaviors in Deep Off-Policy RL"_ (arXiv 2405.18520, 2024) | **Offline-Boosted** ，**Off-Policy** 。OBAC ，。 |
| **Offline + On-policy**  | （，）                                                                                                        | ，——。                                                                               |

## 

[^sutton-barto]: Sutton, R. S., & Barto, A. G. (2018). _Reinforcement Learning: An Introduction_ (2nd ed.). MIT Press.  5.5、5.7、6.4、6.5  off-policy prediction、off-policy control、Sarsa  Q-learning 。MIT Press ：<https://mitpress.mit.edu/9780262039246/reinforcement-learning/>

[^watkins1992]: Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. _Machine Learning_, 8, 279-292. <https://www.gatsby.ucl.ac.uk/~dayan/papers/wd92.html>

[^mnih2015]: Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep reinforcement learning. _Nature_, 518, 529-533. <https://doi.org/10.1038/nature14236>

[^ppo2017]: Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. _arXiv:1707.06347_. <https://arxiv.org/abs/1707.06347>

[^levine2020]: Levine, S., Kumar, A., Tucker, G., & Fu, J. (2020). Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems. _arXiv:2005.01643_. <https://arxiv.org/abs/2005.01643>

[^cql2020]: Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). Conservative Q-Learning for Offline Reinforcement Learning. _NeurIPS 2020_. <https://papers.nips.cc/paper_files/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html>

[^dpo2023]: Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. _arXiv:2305.18290_. <https://arxiv.org/abs/2305.18290>
