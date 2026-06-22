---
search: false
---

# ：（ 10.3）

> 。 [10.3 、 Badcase](./industrial-evaluation)。，。

# 12.5 ：Agentic RL 

 Agentic RL 。，，——、、。，。

2025–2026 ，（ Alibaba、Moonshot、LinkedIn、Bespoke Labs ） Agentic RL 。，****，。

> ****： Agentic RL ，。。

---

## ：

 Agentic RL ，：？

###  API ：

，：****。

> **Moonshot AI**  Kimi-Researcher ，Agent ——，。 **REINFORCE** ， On-policy  [\[\]](https://moonshotai.github.io/Kimi-Researcher/)。

### 

，。

> **Alibaba ** (Tongyi DeepResearch)  API， Wikipedia 。
>
> ****：
>
> 1. ** (WebShaper & AgentFounder)**：， Query ，（MDP）。， **WebShaper**， Wikipedia ； **AgentFounder** （PhD-level）。**** Rollout 。
> 2. ** (rLLM)**：Agentic RL  Rollout （）。 RL （ GPU ），，（GPU）（GPU Idling）。 **rLLM (Ray-based LLM)**  Rollout ，： Worker （ vLLM） Trajectory  Replay Buffer， Trainer （ Megatron/FSDP） Buffer 。
>
> ****：
> ，、，，****。： RL “、”，。 RL  [\[\]](https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/)。

### 

，。

> **Amazon Science**  AppWorld “”：， **72 **（、， API ）。 RL ， Qwen-2.5-32B  39.2%  72%， Claude Sonnet 3.7/4.0。
>
> ****：
>  Agentic RL ： 32B ，。，RL “”，“（Elicit）”。 72 “”，（Exploration），RL （ PPO/GRPO），（Self-Play）。**，，" + RL "  SFT ** [\[\]](https://www.amazon.science/blog/customizing-multiturn-ai-agents-with-reinforcement-learning)。

---

## ：

，。，。

### 

Agentic RL ：**（Rollout）** ，**（Backward）** 。，。

> **LinkedIn **  GPT-OSS（ MoE ） RL ，。， **Attention Sink **：（SGLang  Triton kernel） Attention Sink ，（FSDP  FlashAttention-v2）。 vLLM  FlashAttention ， Sink 。， [\[\]](https://huggingface.co/blog/LinkedIn/gpt-oss-agentic-rl)。

****：，（ GSM8K）， Loss ， Agent 。

---

## ：

 Agentic RL ：， token，。**（Format Collapse）**：

```json
// ：
{"action": "search", "query": "AAPL stock"}

// ：
{"action": "searchsearchAAPL stockAAAAA"
```

。

### ：

，： +1， +1， +5。，。

**（Reward Hacking）** 。，。

> **Bespoke Labs** ，、，，。，、。：**""**（ BFCL  1， 0），， [\[\]](https://www.bespokelabs.ai/blog/improving-multi-turn-tool-use-with-reinforcement-learning)。

：""，，。，Bespoke Labs ，。

### ：

，。，，，。，。

> **Alibaba ** ，，****——，，（）。
>
> ****：
> （Credit Assignment）， **On-policy GRPO（Group Relative Policy Optimization）**，：
>
> 1. **Token-level loss  Leave-one-out **： PPO ，GRPO ，（Relative Advantage）， Token ，。
> 2. **（Conservative Negative Filtering）**：Agent 。 30 ，（），， 20 （CoT）。（ `-1`），RL “”，。，**（Mask out）**，。（Alignment）， [\[\]](https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/)。

### ：KL 

 RLHF/GRPO ， KL 。KL ，。

""""：

- **KL **：，，。
- **KL **：，，。

> **Bespoke Labs**  Qwen2.5-7B-Instruct ， KL  0 ， 300 。：
>
> 1. ** KL **（ 0.001），。
> 2. ****：（ 100 ），。，KL ，"" [\[\]](https://www.bespokelabs.ai/blog/improving-multi-turn-tool-use-with-reinforcement-learning)。

### ：Gamma-decay 

，。

> **Moonshot**  **Gamma-decay Reward**。，：
>
> $$r_i = r \times \gamma^{T-i}, \quad \gamma < 1$$
>
>  $T$ ，$i$ 。：，， [\[\]](https://moonshotai.github.io/Kimi-Researcher/)。

---

## ：

Agentic RL  RL 。、、， 50 ，，。

### 

> **Moonshot**  Kimi-Researcher  **（Context Management）** ，（Long-horizon tasks）（Attention Dilution）“（Lost in the Middle）”。
>
> ****：
>  Agent ，， HTML 、 Observation ， Token 。，LLM （SNR）， 40 “” 1 。
>
> ，Kimi  `context_manager` 。（Step），****：
>
> 1. **（Working Memory）**：（Thought）、（Action）。
> 2. ****：，（Dead-end）。
>
>  Agent “”，。，，， Rollout  50 ，， [\[\]](https://moonshotai.github.io/Kimi-Researcher/)。

---

## ：Agent 

， **Agent （Hallucination）**：， API ，""。Agent ，，。

### Agent 

**。** ，。， `execute_sql`。

**。** ，—— API 、、。：""，。

**。** 。，——，。

**。** "/"，，。 Deep Research Agent ——、URL ""。

### Agent 

，。 Agent ，****：

1.  3 ：， API  → 
2.  4 ：，" API " → 
3.  5 ： → 
4. ：

， RL （ Outcome Reward），""""—— RL ****。，。

### RL 

**。**  AI  CaRR[^carr_industrial]（Citation-aware Rubric Rewards）。（Rubrics），：（1）；（2） URL，， Rubric ；（3） Rubric 。 Rubric  Rubric 。、。

**。** 。（ NLI ），。

**。** """"，。，：

> ****：，，。

```python
def hallucination_aware_reward(answer, tool_results, citations):
    """"""
    reward = base_task_reward(answer)

    # 1. 
    for citation in citations:
        if not verify_citation_exists(citation):
            reward -= 0.5  # ，
        elif not verify_citation_supports(citation, answer):
            reward -= 0.3  # 

    # 2. 
    for claim in extract_claims(answer):
        if has_supporting_evidence(claim, tool_results):
            reward += 0.1  # 
        elif claim_is_verifiable(claim) and not has_supporting_evidence(claim, tool_results):
            reward -= 0.2  # 

    # 3. （）
    if is_complex_question and ("" in answer or "" in answer):
        if not all_claims_supported(answer, tool_results):
            reward += 0.15  # ，

    return reward
```

### 

，****：

**Self-RAG[^selfrag_industrial]** " + "。 RAG ，Self-RAG ****， token（Reflection Token）。，，， [IsRel]（）、[IsSup]（）、[IsUse]（） token ，（Beam Search）。 token 。

**CRITIC[^critic_industrial]** ""。，（、），。，。"→→"，。，CRITIC 。

### 

|      |                 | RL                  |
| ------------ | ----------------------- | --------------------------- |
|  |           |  → reward = 0 |
|      | Schema  +   |  →  reward  |
|      | NLI  +      |  →    |
|      | URL  +  |  →              |

：****。 RL ，。

---

## ：

 Agentic RL 。，（ MoE），。

### MoE 

MoE （ Mixtral、DeepSeek-V3）， RL 。

PPO （ On-policy）， 1。

> **LinkedIn **  GPT-OSS  RL ，MoE （Gating Network）， Token （Expert）， $\log \pi(a|s) \neq \log \pi_{\text{old}}(a|s)$， On-policy 。， `old_log_prob = log_prob.detach()` 。，，—— Attention Sink  [\[\]](https://huggingface.co/blog/LinkedIn/gpt-oss-agentic-rl)。

### MoE 

MoE  RL ， GPU 。 Token ""， GPU ， GPU 。

> **Salesforce**  SFR-RL  ** RL（Pipelined Synchronous）** ： GPU  Rollout  Training ， GPU 。， MoE ， **Least-Loaded Expert Parallelism** 。 VERL（FSDP + Context Parallelism） 250 ， 16  H200  120B  MoE  [\[\]](https://www.salesforce.com/blog/efficient-rl-training-agentic-era/)。

### 

，RL ****，。 RL 。

> **Amazon Science** ：32B  RL ，（Rollout），。，——，RL 。，（Distillation）， RL  [\[\]](https://www.amazon.science/blog/customizing-multiturn-ai-agents-with-reinforcement-learning)。

### 

，， RL 。

 SFT ：**SFT-RL ****Pure-RL **。

> **SFT-RL （）**：**Alibaba ** Tongyi DeepResearch  **CPT → SFT → RL** 。（CPT）； SFT ； RL 。：（ API 、），SFT/RM ****。， RL  [\[\]](https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/)。

> **Pure-RL （）**： **DeepSeek-R1-Zero** 。（、、），** SFT **， Base Model 。 RL ，（CoT）、、。 SFT （Bias-free），。

 Agentic RL ，****。

---

##  {#tricks}

：

|                    |                                                              |              |
| ---------------------- | -------------------------------------------------------------------- | ---------------- |
|        |                                                  | Alibaba          |
|          | （ 72 ） RL                    | Amazon           |
|        | （ Attention Sink ） | LinkedIn         |
|      | （）；             | Bespoke Labs     |
|        |  KL （ 0.001）；       | Bespoke Labs     |
| （） |  Gamma-decay ，                    | Moonshot         |
|                | ，       | Alibaba          |
|        | ，                       | Moonshot         |
| MoE    |  RL + Expert Parallelism；16  H200  120B MoE     | Salesforce       |
| MoE          |  MoE  On-policy ； | LinkedIn         |
|      |  RL； CPT → SFT → RL       | Amazon / Alibaba |

##  {#references}

- Zhu J, Sang H, et al. "[Unlocking Agentic RL Training for GPT-OSS: A Practical Retrospective](https://huggingface.co/blog/LinkedIn/gpt-oss-agentic-rl)." Hugging Face Blog, 2026.
- Zhuang R, Vu T, et al. "[Improving Multi-Turn Tool Use with Reinforcement Learning](https://www.bespokelabs.ai/blog/improving-multi-turn-tool-use-with-reinforcement-learning)." Bespoke Labs Blog, 2025.
- Moonshot AI. "[Kimi-Researcher: End-to-End RL Training for Emerging Agentic Capabilities](https://moonshotai.github.io/Kimi-Researcher/)." 2025.
- Tongyi DeepResearch Team. "[Tongyi DeepResearch: From Chatbot to Autonomous Agent](https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/)." 2025. [GitHub](https://github.com/Alibaba-NLP/DeepResearch)
- Salesforce AI Research. "[Building Efficient RL Training for the Agentic Era](https://www.salesforce.com/blog/efficient-rl-training-agentic-era/)." 2026.
- Subramanian S, Xu P, Wang Y. "[Customizing Multiturn AI Agents with Reinforcement Learning](https://www.amazon.science/blog/customizing-multiturn-ai-agents-with-reinforcement-learning)." Amazon Science Blog, 2026.

[^carr_industrial]: Zhang J, Lv X, Feng L, Hou L, Li J. "[Chaining the Evidence: Robust Reinforcement Learning for Deep Search Agents with Citation-Aware Rubric Rewards](https://arxiv.org/abs/2601.06021)." arXiv, 2026.

[^selfrag_industrial]: Asai A, et al. "[Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)." ICLR 2024.

[^critic_industrial]: Gou Z, et al. "[CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing](https://arxiv.org/abs/2305.11738)." ICLR 2024.

---

 Agentic RL 。 [10.3 ：、 Badcase](./industrial-evaluation)， benchmark、 badcase  Agent 。
