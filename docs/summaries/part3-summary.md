# Part 3:  — 

##  Part ？

。 RL ，。，：

- **RLHF **：SFT → RM → RL ， $R = R_{\text{RM}} + \alpha R_{\text{format}} + \beta R_{\text{length}}$，，RLAIF  AI 。
- **DPO **：$r(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$。，。
- **DPO **：， RL 。， Critic  RM。
- **DPO **：KTO /（），SimPO ，ORPO  SFT 。
- **GRPO **：$A_i = \frac{r_i - \text{mean}(r_1, \ldots, r_k)}{\text{std}(r_1, \ldots, r_k)}$。 prompt  $k$ ， $z$-score  Critic。 1 。
- **DAPO **：Clip-Higher（）、Token （ token ）、（ prompt）、Overlong 。
- **RLVR**：、（、）。DeepSeek-R1-Zero  RLVR 。
- **Agentic RL**：ORM（）vs PRM（）， RL "SFT  + RL "。

。

##  8 ：RLHF ——

### 

 RLHF ，" RM "。：

$$R_{\text{total}} = R_{\text{RM}} + \alpha R_{\text{format}} + \beta R_{\text{length}} + \gamma R_{\text{correctness}}$$

：Sequence-level（）、Step-level（， PRM）、Token-level（ token ）。

### ：

（Reward Hacking）：、、。、、。KL  $-\beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ 。

### RLAIF： AI 

RLAIF 。Constitutional AI ，： →  →  →  AI  → 。

##  9 ：（DPO + GRPO + RLVR）

###  RLHF  DPO：

 RLHF ：（Reward Model）， PPO ， KL 。，。

DPO ： KL  RL 

$$\max_\pi \mathbb{E}_{x,y \sim \pi}[r(x,y)] - \beta D_{\text{KL}}(\pi \| \pi_{\text{ref}})$$



$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)$$

，：

$$r(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

 $Z(x)$  $x$  $y$， Bradley-Terry  $P(y_w \succ y_l|x) = \sigma(r(x,y_w) - r(x,y_l))$ ，$Z(x)$ 。 DPO ：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

### GRPO： Critic

PPO  Critic  $A(s,a)$。 LLM ， Critic ，。GRPO（Group Relative Policy Optimization）：

 prompt $x$， $k$  $y_1, y_2, \ldots, y_k$， $r_1, r_2, \ldots, r_k$。：

$$A_i = \frac{r_i - \text{mean}(r_1, \ldots, r_k)}{\text{std}(r_1, \ldots, r_k)}$$

 PPO  Critic  $A = Q - V$ ——""—— GRPO  Critic 。

### DAPO： GRPO 

**Clip-Higher** ，。**Token **  token ，。****  prompt，。**Overlong ** 。

### RLVR：

GRPO/DAPO  RM——。、，：，。DeepSeek-R1-Zero ， SFT， RLVR ， chain-of-thought 。

##  10 ：Agentic RL——

### 

 RL ""， Agent 。**ORM**（Outcome Reward Model），。**PRM**（Process Reward Model），。

###  RL 

Web Agent  Code Agent  Agentic RL 。"SFT  + RL "：， RL 、。

## 

Part 3 ：RLHF  4  → DPO  2  → GRPO  1  → RLVR  RM。""，。

，RL """"""——，RL 。

> ****：[Part 4: ](/chapter11_vlm_rl/intro)
