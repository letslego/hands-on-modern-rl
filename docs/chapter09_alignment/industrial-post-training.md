# 9.7 

 DPO、GRPO、RLVR ，，：、SFT、、、 rollout、、、。 2026-05-06 ，：、、、SFT/RL 、。

## 

### MiniMax

> ****：[MiniMax M2.1: Post-Training Experience and Insights for Agent Models][^minimax_m2_1]、[MiniMax-M1][^minimax_m1]、[WebExplorer][^minimax_webexplorer]

MiniMax ：M2.1  agent 、MiniMax-M1  RL scaling、WebExplorer  agent  RL。 agent、 agent、 agent  reasoning model，“ + reward”。

#### 1. M2.1  SWE Scaling： GitHub  RL 。

****：， GitHub  agent。 issue、PR  commit ， RL ；、、、，“ ->  -> reward”。

****：

- ****： GitHub merged PR、commit、issue、 diff 。
- ****：SWE-Resolve  bug ；SWE-Test ，、；SWE-Review 。
- ****：checkout ，，， Docker / sandbox， patch 。
- ****： original problem description、test-case reward、runnable environment ， SFT  RL 。
- ****：M2.1  10+ 、10,000+ runnable PRs  140,000+ variable tasks。

****：

- **SWE-Resolve**： reward ，、、。
- **SWE-Test**： patch 、patch ， bug。
- **SWE-Review**：，， LLM  review ， hallucination rate。

****“”。 GitHub 、、。

**Multi-scaffold**  M2.1 。scaffold  agent ，、、/、。 SFT  RL  ReAct loop ， loop 。MiniMax  multi-scaffold rejection sampling  SFT ， RL  scaffold  rollout。

**** scaffold： ReAct、plan-then-edit、test-driven loop； SWE  scaffold， SFT， RL  scaffold，。

#### 2. M2.1  AppDev： Agent-as-a-Verifier。

****：“”。//，，、、、。

****：MiniMax  experts-in-the-loop。、、Android、iOS  prompts、meta-queries、rubric-based rewards 。；，。

**Reward **：

- **Execution level**：、、。
- **Interaction level**： Playwright 、、，。
- **Visual level**：，、、。

** LLM-as-a-judge **：judge ， agent  sandbox 。

**** Todo App /  / ：，； `npm install && npm run dev`； Playwright  5-10 ； rubric judge 。

#### 3. WebExplorer： agent 。

****： web agents  BrowseComp、GAIA、WebWalkerQA、FRAMES ，、、。； query evolution ；。

****：WebExplorer  model-based exploration ， iterative long-to-short query evolution 。

**WebExplorer-QA **“”“”：

- ****： Wikipedia seed entities ， prompt  BrowseComp-en  QA exemplars， seed entity  search / browse，。，、、。
- ****： long-to-short query evolution。，，、、。 5  evolution， 40K WebExplorer-QA。
- ****：，，。

** recipe** ：

- ** SFT**： search / browse ，。 Qwen3-8B ， 13K SFT ，batch size 32，learning rate 1e-5， 4 epochs。
- **GRPO **：， QA ，。 12K  GRPO， 8 rollouts，batch size 64，learning rate 1e-6， 128K、 100。

**Reward **：`R = 0.2 * R_format + R_correct`。

- **`R_format`**：、、。
- **`R_correct`**： DeepSeek-V3  judge， ground-truth answer 。
- ****：web agent RL ，， search / browse ；，。，。

****： 500-1000  Wikipedia seed entities； seed  5-10  QA；、、； 2-3  query compression，、、； URL； 1K-5K  SFT， 7B/8B  search/browse ； 1K QA  GRPO，reward  + LLM judge / exact match。

****：、、。

#### 4. MiniMax-M1： RL 。

****：M1  agent ， test-time compute scaling 。 CoT  64K、80K、100K token ， attention  RL 。

** RL**：M1  hybrid MoE + Lightning Attention  1M  40K/80K thinking budget， CISPO  RL。CISPO  clip importance sampling weights， clip token updates； token ，。

** curriculum**：

- ****： MiniMax-Text-01 ， STEM、、、reasoning、。
- **SFT **： CoT ， RL 。
- **RL **：、、SynLogic  41 、SWE-bench  sandbox； QA、， reward model 。
- ****： 80K ，， 40K  48K、56K、64K、72K、80K。
- ****： perplexity 、99 。
- ****： 40K 、、、 synthetic reasoning 。

**** 512 H800，：；， RL ； reasoning ， RL 。

###  Qwen / 

> ****：[Qwen2.5 Technical Report][^qwen2_5]、[Qwen2.5-Math][^qwen2_5_math]、[QwQ-32B][^qwq_32b]、[Qwen3][^qwen3]、[Qwen3-Coder][^qwen3_coder]、[Qwen3-Coder-Next][^qwen3_coder_next]、[Tongyi DeepResearch][^tongyi_dr]

Qwen ： instruct 、、reasoning RL、agentic coding / deep research。“”“”， GRPO 。

#### 1. Qwen2.5： instruct 。

Qwen2.5  post-training  SFT。、、、、、， RL。 motivation ：base model ，； SFT “”。 Qwen2.5  instruction following、、、。， SFT ， JSONL； eval，、JSON 、、、。

#### 2. Qwen2.5-Math： CoT / TIR 。

 motivation  CoT 、， Qwen2.5-Math  Chain-of-Thought  Tool-Integrated Reasoning。， CoT / TIR ，、Python executor、majority voting  reward model 。Qwen2.5-Math-RM-72B  reward model， rejection sampling， RL。

TIR 。 token 、Python 、； executor ， mask  executor output token  loss。：。RL  GRPO， verifier / reward model ， KL 。 Qwen2.5-Math ，，，。

： GSM8K / MATH / OlympiadBench ； 4-16  CoT  Python TIR ； parser、`sympy`、Python  majority vote ； SFT； 8 ， verifier  0/1 ， GRPO。 TIR  loss mask，、。

#### 3. QwQ-32B： outcome-based reward  RL。

QwQ-32B  RL ： cold-start checkpoint ， math  coding RL， reward model， accuracy verifier 。 reward ； reward 。“”。

 general capabilities RL， general reward model  rule-based verifier， instruction following、 agent performance，。 recipe：，。，， 5%-20% //， math、code、chat、agent  eval 。

#### 4. Qwen3：thinking / non-thinking 。

Qwen3 “ GRPO”，。 long-CoT cold-start  reasoning RL：cold-start 、；reasoning RL  query-verifier pairs  GRPO。 query-verifier pair ： cold-start ； cold-start ；；。 3,995  query-verifier pairs， batch、 rollout、off-policy ， entropy 。

“”。Qwen3  reasoning path  reasoning path ， thinking  non-thinking； general-domain RL 、、。： `<think>`  CoT； verifier  RL；、，；、、。

#### 5. Qwen3-Coder / Tongyi DeepResearch： RL  RL。

Qwen3-Coder  repository-level action：、 bug、 patch、、、。reward 、、、issue  patch 。Tongyi DeepResearch  search / read / synthesize ：，、、、。 Qwen  agent “prompt -> answer”“environment episode -> verified outcome”。 SWE-bench Lite  QA：， SFT， answer judge  RL。

### Moonshot Kimi

> ****：[Kimi k1.5][^kimi_k1_5]、[Kimi K2][^kimi_k2]、[Kimi-Researcher][^kimi_researcher]

Kimi  reasoning scaling、 agentic model、research agent。 k1.5  Kimi-Researcher：“ RL ”，“ agent  RL ”。

#### 1. Kimi k1.5： MCTS / value function / process RM， RL。

k1.5  motivation  test-time compute scaling。，：policy ，reward  outcome，policy optimization  KL  reward 。 MCTS、value function、process reward model ：， rollout 。

，k1.5 。、、 rule / execution verifier；、、 reward model  judge。 prompt ，reward ， policy mirror descent 。， reward ；“”。

#### 2. ： overthinking， max tokens。

k1.5  overthinking： CoT ， token ，。 length reward ，：；，。“” reward， generation max length。

length reward  warm up。，，，；，。： 30%-50% RL steps  correctness reward；， 0.1-0.3 ，； response length，。

#### 3. Long-to-short：，。

k1.5  long-to-short  length reward 。，；、，。“”：，。 CoT ， concise solution， SFT / DPO， verifier 。

#### 4. Kimi K2：agentic intelligence 。

K2  open agentic intelligence， benchmark，、。、、、、。K2  agent ，、、、verifier  judge，， SFT / RL。

#### 5. Kimi-Researcher： agent  reward 。

Kimi-Researcher 。 research episode：，/，，，，。 reward “”，、、、、。： 200-500 ；； judge  evidence coverage、citation correctness、answer faithfulness、redundant-search penalty； SFT ， episode-level reward  GRPO / DPO。

###  Seed / Doubao

> ****：[Seed1.5-Thinking][^seed1_5_thinking]、[VAPO][^vapo]、[DAPO][^dapo]、[DAPO GitHub][^dapo_github]、[UI-TARS][^ui_tars]、[UI-TARS GitHub][^ui_tars_github]、[UI-TARS-2][^ui_tars_2]、[Seed Prover 1.5][^seed_prover]、[Seed1.8][^seed1_8]

 Seed ：reasoning RL ， GUI / prover  agent 。DAPO、VAPO、UI-TARS-2 ，“ rollout ”。

#### 1. Seed1.5-Thinking：reasoning model 。

Seed1.5-Thinking  RL 、。：，， verifier。SFT  CoT ，RL  outcome reward 。 DeepSeek-R1、Qwen3 ， Seed 、 advantage 。

#### 2. DAPO： GRPO 。

DAPO  motivation  reasoning RL ， GRPO ，、clip、。 GRPO 。

Dynamic Sampling “” prompt。， reward ，advantage 。DAPO ， batch  advantage ，。Clip-Higher  PPO clip ： CoT  token ，，； `eps_clip_high` ， low 0.2、high 0.28。

Token-Level Policy Gradient 。 sequence-level loss  CoT  token  token ，；DAPO  token ， token 。Overlong Reward Shaping ：，， reward ；、 shaping。

： Qwen2.5-7B/32B base、AIME/MATH 、 8-16  rollout； GRPO  baseline； dynamic sampling、clip-higher、token-level loss、overlong shaping； prompt 、entropy、、AIME pass@1 。DAPO  ablation 。

#### 3. VAPO：value model ， CoT  advantage 。

VAPO  value-model-based RL。 CoT ，GAE  reward  token， advantage 。 ablation ： decoupled GAE ；Length-Adaptive GAE  GAE ， credit；token-level policy gradient ；positive-example LM loss ；group sampling  prompt、 repetition 。 `epsilon_low=0.2`、`epsilon_high=0.28`、positive LM loss weight 0.1、512 prompts  16 samples 。

VAPO  value model，： GRPO  value baseline  PPO/VAPO ， advantage 、reward 。 SOTA， RL  credit assignment 。

#### 4. UI-TARS：GUI agent 。

UI-TARS /、，、、 GUI action。 action trace 。，，、VLM 。Reflection tuning ：，， DPO 。

#### 5. UI-TARS-2： RL、。

UI-TARS-2  motivation  GUI-only agent ：、、、。 hybrid GUI environment， GUI、file system、terminal  sandbox ， rollout  RL。 flywheel ：， SFT， continual pre-training ；，、。

 reward ：，，，，，，。 MiniWoB / BrowserGym / OSWorld ： action schema； reset、observe、step、success check； 200  SFT； rollout + success reward  RL； reflection。

#### 6. Seed Prover 1.5： agentic RL。

 theorem prover，。 tactic、 lemma、；reward  proof 、、 lemma 。 agent RL ：， episode。Seed1.8  reasoning、multimodal、tools  generalized agent ，“”“”。

### DeepSeek

> ****：[DeepSeekMath][^deepseek_math]、[DeepSeek-R1][^deepseek_r1]、[DeepSeek-V3.2][^deepseek_v3_2]

DeepSeek  GRPO / RLVR 。DeepSeekMath  critic-free ，R1  reward ，V3.2  agentic task synthesis。

#### 1. DeepSeekMath：GRPO 。

DeepSeekMath-RL  DeepSeekMath-Instruct 7B ， 144K  GSM8K、MATH  CoT  RL。， reward model / ， advantage。 PPO  critic/value model，。 policy learning rate 1e-6、KL coefficient 0.04、 64 、max length 1024、batch size 1024， exploration  policy update。

GRPO ：“”，“”。，、、、。 7B  SFT 、MATH 、 8-16  rollout、 parser ； reward ， KL ， GSM8K/MATH 。DeepSeekMath ，SFT ， RL  out-of-domain reasoning 。

#### 2. DeepSeek-R1-Zero： base model  RL。

R1-Zero  motivation  CoT  SFT。 base model  RL， accuracy reward  format reward：/， reward 。、、、， reasoning pattern  outcome reward pressure 。

R1-Zero ：、、。“ RL ”， recipe。， R1-Zero ： base model ，/，；，、、。

#### 3. DeepSeek-R1：cold-start + reasoning RL + rejection sampling + final RL。

R1 。 cold-start 、； reasoning-oriented RL，、、； rejection sampling， SFT ，、、，； RL， helpfulness、harmlessness  reasoning。

“，”。 RL /，、；rejection sampling  final RL 。 1K cold-start  CoT、20K  RL、 RL  10K  SFT ，/ DPO  GRPO。 math/code、、、。

#### 4. DeepSeek-V3.2： verifier  agentic verifier。

V3.2  agentic tasks。，、、 episode。reward ，、、、。 MiniMax M2.1、UI-TARS-2、LongCat ：RLVR  math verifier  software / browser / GUI / tool verifier。

###  Z.ai / GLM

> ****：[GLM-4.5][^glm_4_5]、[GLM-5][^glm_5]

GLM  ARC：Agentic、Reasoning、Coding。GLM-4.5  MoE ；GLM-5  recipe ：multi-task SFT ， Reasoning RL、Agentic RL、General RL ， RL 。

#### 1. GLM-4.5：hybrid reasoning  expert model iteration。

GLM-4.5  thinking  direct response 。 motivation  Qwen3 ：， CoT。 expert model iteration  RL  agentic、reasoning、coding。expert iteration “，”；RL 、、 agent benchmark 。

#### 2. GLM-5：Reasoning RL -> Agentic RL -> General RL。

GLM-5  progressive alignment： multi-task SFT， interleaved thinking modes； reasoning RL； agentic RL； general RL 。Reasoning RL 、、 outcome verifier ，。Agentic RL 、、，“ ->  ->  -> ”。General RL 、、、，。

#### 3.  agent RL： generation  training 。

GLM-5  asynchronous RL infrastructure， slime  rollout interface。 agent rollout ：，、、。 PPO/GRPO  episode。GLM-5  rollout generation、、verifier ，。slime  server-based rollout execution  multi-turn loop、tool invocation、environment feedback handling、verifier-guided branching，。

#### 4. On-policy cross-stage distillation：。

 RL ：Reasoning RL  Agentic RL  General RL 。GLM-5  on-policy cross-stage distillation ：，。：MATH/ GRPO  reasoning ；SWE-bench Lite  RL  agentic ； DPO/GRPO， distillation，。

###  Hunyuan

> ****：[Hunyuan-T1][^hunyuan_t1]、[Hunyuan-A13B][^hunyuan_a13b]、[Hunyuan-A13B-Instruct Model Card][^hunyuan_a13b_instruct]

 T1  reasoning RL  A13B  fast/slow thinking instruct model。A13B  MiniMax/MiMo ， T1 ， reasoning RL 。

#### 1. Hunyuan-T1： post-training compute  RL。

T1  post-training  96.7%  reinforcement learning，。、、、 world science and reasoning problems， ground-truth feedback。 reward ：、、、 judge。 reward ， RL， reasoning-heavy RL。

#### 2. curriculum、context expansion  token efficiency。

T1 ：，，， token。 MiniMax-M1 / Qwen3 ：、 reward ，。/， CoT + ， CoT + ，/。

#### 3. data replay、periodic policy reset  reward。

T1  data replay  periodic policy resetting， 50%。 RL ：data replay ；policy reset ， checkpoint / reference。 self-reward + reward model： T1-preview  self-reward ，， reward model，。 replay buffer，； N  entropy、、 eval， reference。

#### 4. Hunyuan-A13B：fast / slow thinking 。

A13B-Instruct  slow-thinking ， `enable_thinking=False`  CoT。： `<think>` ，。，。 prompt ：，；SFT  RL / DPO “ thinking”，。

###  ERNIE

> ****：[ERNIE 4.5 Technical Report][^ernie_4_5]、[ERNIE 5.0][^ernie_5_0]

ERNIE 4.5 ：LLM post-training  SFT + RL，RL  Progressive RL  Unified Preference Optimization；VLM post-training  SFT +  reasoning RL。ERNIE 、 reward、“”。

#### 1. SFT：， RL。

ERNIE 4.5  SFT 、、、、、。，。 LLM ，SFT ； VLM ， SFT 、， thinking / non-thinking 。 RL，， SFT  perception 。

#### 2. Unified Rewarding System： rule、sandbox、RDRM、GRM 。

ERNIE 4.5  rule-based reward、RLLM、sandbox、RDRM、checklist-aware verifier、GRM、DRM 。 reward ：； sandbox ； generative reward model；/ checklist verifier； discriminative reward model。 domain normalization， reward 。ERNIE ： verifier / RM； reward ；。

#### 3. Progressive RL：Logic RL -> Reasoning RL -> General RL。

ERNIE 4.5  LLM RL  Stage 1 Logic RL、Stage 2 Reasoning RL、Stage 3 General RL。Logic RL 、；Reasoning RL 、、；General RL 、。 GLM-5 / Qwen3 “、”。： 2K /， 10K /， 10K ；。

#### 4. UPO： RL 。

Unified Preference Optimization  reasoning tasks  non-reasoning tasks ，reward-format、domain normalization、informative prompt filtering 。/ 0-1 reward、、。UPO ： reward normalization； prompt； reward source ； reward ，。

#### 5. ERNIE 5.0：。

ERNIE 5.0 、、、。 reward ： reward、 reward、 reward、 reward 。 JSON， eval、 eval  eval， SFT/RL。

###  StepFun

> ****：[Step3][^step3]、[STEP3-VL-10B][^step3_vl_10b]、[Step-DeepResearch][^step_deepresearch]

StepFun  reasoning  deep research agent。STEP3-VL-10B  10B VLM  scaled post-training ；Step-DeepResearch  agent 。

#### 1. STEP3-VL-10B：fully unfrozen ， 1K+ RL iteration 。

 motivation ，。 1.2T multimodal tokens 、， perception encoder  Qwen3-8B decoder ；post-training  1K iterations  reinforcement learning。：VLM  RL ，、 reward 。

#### 2. RLVR + RLHF：。

、OCR 、、、/ RLVR：，/。、、 RLHF / judge reward。： MathVista、ChartQA、OCR-VQA、 exact / numeric verifier； judge  helpfulness、faithfulness、detail、safety 。 reward ，。

#### 3. PaCoRe：，。

Parallel Coordinated Reasoning  test-time compute。，“”“”。PaCoRe ，。：，。 self-consistency ：， verifier / judge ， SFT “ ->  -> ”。

#### 4. Step-DeepResearch：。

Deep research agent 、、、、。SFT  research trajectories， query、、；RL  reward  answer correctness、citation existence、evidence support、source coverage、redundant search penalty、final report structure。 300 、 API、 citation checker，， episode-level reward。

###  LongCat

> ****：[LongCat-Flash-Thinking-2601][^longcat_flash]

LongCat-Flash-Thinking-2601 “agent RL ”。 reward ，、、 heavy thinking。

#### 1. ：。

LongCat  motivation  agent ， prompt、。 20+ 、：， 60+ 、 schema、。、、、。“”“”。

。，，“、”。LongCat ：；； BFS ，；； 20，。：，，。

#### 2. ：。

LongCat  RL /“ RL ”。，，；，，。；，。 5 ：、、、、；，。

#### 3. DORA： RL。

Agent rollout ， GPU。DORA ，；。 Rollout Manager  Rollout Controller， rollout ，。 PyTorch RPC， CPU 。

 5600  MoE，DORA  Prefill-Decode  KV-cache 。PD  prefill  decode ， prefill  decode；KV-cache  chunk 、、 CPU ，。： rollout ，。 2-4 ，。

#### 4. ：。

LongCat 、、、、、，。reward ，、、、。 10%-30% ，、； clean success rate  noisy success rate 。

#### 5. Heavy Thinking：。

LongCat  CoT ，/，、。 agent ，，。 3-5 ， verifier / judge ，。“ ->  -> ” SFT / RL。

###  Ling / Ring

> ****：[Ling-1T][^ling_1t]、[Ring-1T][^ring_1t]

#### 1. ：， recipe 。

Ling / Ring ， DeepSeek-R1、Qwen3  MiniMax M2.1 。“”，。

#### 2. ：deep thinking 。

：，trillion-scale MoE  deep thinking / insight  post-training ，；，。

#### 3. ：fast / slow thinking 。

 fast/slow thinking ：、、，；， eval  token cost。

###  Pangu

> ****：[Pangu Ultra][^pangu_ultra]、[Pangu Pro MoE][^pangu_pro_moe]、[][^pangu_news]

#### 1. ：。

Pangu 、MoE ， R1/Qwen/MiniMax 。

#### 2. ： recipe 。

 recipe ： Ascend NPU ，post-training ， MoE 、、。

#### 3. ： eval。

“”： reasoning 、、、。

### 01.AI Yi

> ****：[Yi-Lightning][^yi_lightning]

#### 1.  RLHF 。

Yi-Lightning  LLM ：pre-training  SFT  RLHF， multi-stage training、synthetic data construction、reward modeling， RAISE  pre-training、post-training  serving。 agent  recipe，“”。

#### 2. ：SFT -> RM -> PPO / DPO。

：/ SFT； prompt ， judge  reward model； PPO/DPO， Chinese、Math、Coding、Hard Prompts  safety。

#### 3. ： benchmark。

Yi-Lightning ： benchmark ，。

### InternLM /  AI Lab

> ****：[InternLM2][^internlm2]

#### 1.  RLHF ：。

InternLM2  RLHF 。 CoT RLVR，、SFT、reward modeling  online RLHF。

#### 2. COOL：，。

COOL（Conditional Online RLHF）：，，。、，。

#### 3. ： domain / style / safety 。

： domain / style / safety ； reward model ； RLHF  prompt  reward； helpfulness、harmlessness、verbosity 。

****： verifier， RL ，。

###  Baichuan  360 

> ****：[Baichuan 2][^baichuan2]、[360Zhinao][^zhinao]

#### 1. Baichuan2： SFT -> RM -> PPO。

Baichuan2  SFT -> RM -> PPO 。SFT  base model ；RM ， reward model；PPO  RM  KL 。 InstructGPT /： RLVR ，SFT/RM/PPO 。

#### 2. 360Zhinao：RM 。

360Zhinao 。RM  PPO ， judge、：，，， SFT。

#### 3. ：rejection sampling SFT + DPO。

 4 ， judge/RM ， top-1  rejection sampling SFT， bottom/top pair  DPO。 agent RL ，。

###  Skywork   MiMo

> ****：[Skywork-OR1][^skywork_or1]、[MiMo][^mimo]、[MiMo-VL-Miloco][^mimo_vl]

Skywork-OR1  MiMo “ /  RL”。 frontier lab ， entropy collapse、、reward 。

#### 1. Skywork-OR1： R1-Distill  RL， entropy collapse。

Skywork-OR1  DeepSeek-R1-Distill 。 CoT， RL ，entropy 。 pipeline  ablation  entropy dynamics ， premature entropy collapse 。 32B  57.8%  72.8%，7B  43.6%  57.5%，、。

 entropy， reward。 R1-Distill-7B / RL； token entropy、response length、pass@1、 n-gram、；、KL、clip、。 reward  entropy ，。

#### 2. MiMo：7B reasoning model  post-training  130K 。

MiMo-7B  post-training  130K verifiable mathematics and programming problems  RL。 verifier；。 test-difficulty-driven code reward， reward ：/，。Strategic data resampling ，。

MiMo ： 80K  50K ， 5K/2K ； Math-Verify / parser ； easy/medium/hard tests，reward ； RL 、、，/，。 7B ，， rollout 。

#### 3. MiMo-VL-Miloco：。

MiMo-VL “ +  +  RL”，。 STEP3-VL ： perception  reasoning ；reward 、。/OCR/ RLVR ，。

### 、、

> ****：[Kwai Keye-VL][^keye_vl]、[SenseNova U1][^sensenova_u1]、[Spark X1][^spark_x1]

#### ：。

（）、（）（）， recipe 。“”，。

****：， VLM、、、； SFT/RL 。

---

## 

### OpenAI

> ****：[InstructGPT][^instructgpt]、[GPT-4][^gpt4]、[o1][^o1]、[o3/o4-mini][^o3_o4_mini]、[o3 Operator][^o3_operator]、[GPT-4.5][^gpt4_5]、[GPT-5][^gpt5]、[GPT-5.1][^gpt5_1]、[GPT-5.4 Thinking][^gpt5_4]、[GPT-5.5][^gpt5_5]、[GPT-5.5 Instant][^gpt5_5_instant]、[GPT-5-Codex][^gpt5_codex]、[GPT-5.1-Codex-Max][^gpt5_1_codex_max]、[GPT-5.2-Codex][^gpt5_2_codex]

OpenAI ：InstructGPT  RLHF、o-series  reasoning / deliberation、 deliberative alignment， Codex / Operator  agent 。 recipe，。

#### 1. InstructGPT：RLHF 。

InstructGPT 。 demonstration SFT：， base model 。 reward modeling： prompt ，， reward model 。 PPO： reward model  policy ， KL penalty  policy  SFT 。：SFT “”，RM “”，PPO “prompt + on-policy samples”。

 5K  SFT； 1K prompt  4 ， pairwise preference  RM； PPO / DPO / IPO 。 reward model ， LLM judge  helpfulness、truthfulness、toxicity、verbosity  instruction following， RM 。

#### 2. GPT-4  o-series：reasoning 。

GPT-4  post-training  safety；o1/o3/o4-mini ： deliberation，。 action “ token”，、、/、、。reward 、、、。

：，；； execution result；reward 、、。 SFT ， RL。 o-series ，。

#### 3. Deliberative alignment：。

OpenAI ，。；deliberative alignment  policy spec、：，，。 prompt、， SFT ， preference/RL “”。

#### 4. Operator / Codex：agent 。

Operator  Codex  browser / software engineering episode。coding agent 、、patch verifier、lint、；browser agent 、、。GPT-5-Codex  RL ， PR 、、；GPT-5.1-Codex-Max  agentic coding， compaction  token ；GPT-5.2-Codex  SWE-Bench Pro、Terminal-Bench 2.0、Windows 、。 SWE-bench Lite：checkout ， issue，， tests，reward  + patch ； MiniWoB / BrowserGym： DOM/，，reward 。

### Anthropic

> ****：[Constitutional AI][^constitutional_ai]、[Anthropic CAI overview][^anthropic_cai]、[Claude 4 System Card][^claude4]、[Claude Sonnet 4.5][^claude_sonnet_4_5]、[Claude Opus 4.5][^claude_opus_4_5]、[Claude Opus 4.6][^claude_opus_4_6]

Anthropic  Constitutional AI 。 Claude 4  recipe， Constitutional AI 。

#### 1. Constitutional AI：， AI feedback 。

 RLHF 。Constitutional AI ， constitution；supervised phase ，， constitution ， SFT 。preference phase ，AI  constitution ，， preference model， RL  policy。 RLAIF：。

： 20-50 ///； prompt ；； SFT；， DPO  reward model。 over-refusal，。

#### 2. Claude ：。

Claude 4  reward hacking、sabotage、sycophancy、alignment faking、hidden objectives、jailbreak、extended thinking 。 RL ， adversarial evaluation。 reward ，、、。

#### 3. Extended thinking 。

，“”。，。 reward 、、、、。：，；reward 。

### Google DeepMind

> ****：[Gemini 1.5][^gemini_1_5]、[Gemini 2.5][^gemini_2_5]、[Gemini 2.5 Deep Think][^gemini_2_5_deep_think]、[Gemini 2.5 Computer Use][^gemini_2_5_computer_use]、[Gemini 3.1 Pro][^gemini_3_1_pro]、[Gemma 3][^gemma_3]

Google DeepMind ，：、、、reasoning 。Gemini / Gemma “”。

#### 1. Gemini 1.5 / 2.5： evidence grounding。

 context window 。 token、、、，。 needle-in-haystack、 QA、、。reward ，、、。

#### 2. Deep Think： + ， CoT。

Gemini 2.5 Deep Think  test-time compute scaling ：，。 LongCat heavy thinking、self-consistency 。 reward “”“”：，。/ 5 ，verifier ，“ + ”。

#### 3. Computer Use：GUI 。

Gemini Computer Use ：/，、、，。reward 、、、、、。 BrowserGym / OSWorld： task  reset、observe、step、success check ； SFT ， RL 。

#### 4. Gemma： distillation + targeted post-training。

Gemma ：，、、、 post-training。： frontier  RL ，、、 targeted preference optimization 。

### Meta Llama

> ****：[The Llama 3 Herd of Models][^llama3_herd]

Llama 3 Herd  chat model 。，、SFT、reward model、rejection sampling、preference optimization、。

#### 1. SFT 。

Llama  SFT “ instruction JSON”。、、、、、， eval。、、、、/。，。

#### 2. Rejection sampling： RM /  SFT 。

 prompt ， reward model、 verifier  judge ， SFT。 SFT  RL ：，。/ verifier；/ RM / judge。， prompt  4-8 ， top-1， top/bottom pair  DPO。

#### 3. Preference optimization  safety 。

Llama ，、SFT、 RM、、。Preference optimization ，， truthfulness、、、helpfulness 。： agent ， SFT、RS、DPO/RLHF、。

### Microsoft Phi

> ****：[Phi-4][^phi_4]、[Phi-4-reasoning][^phi_4_reasoning]

Phi-4-reasoning  reasoning。，、teachable prompts  outcome-based RL， 14B 。

#### 1. ：teachable prompts 。

，、、。、、， RL ， rollout。 5K-20K ， SFT ， RL。

#### 2. SFT ， RL 。

Phi-4-reasoning ： synthetic reasoning traces  SFT，； outcome reward  RL，。 average response length， RL 。 Phi/Qwen 7B-14B、MATH/GPQA 、 CoT，SFT  8 ， verifier  GRPO，。

### NVIDIA Nemotron

> ****：[Nemotron-4 340B][^nemotron_4]、[Llama-Nemotron][^llama_nemotron]、[Llama Nemotron Ultra][^nemotron_ultra]、[Nemotron Agent Blog][^nemotron_agents]、[Nemotron-H][^nemotron_h]、[Nemotron 3][^nemotron_3]

NVIDIA Nemotron ：、、reward、。Nemotron-4 340B  synthetic data、preference data、reward model；Llama Nemotron  reasoning、tool use、RAG、instruction following 。

#### 1. Nemotron-4：alignment 。

 instruct ， synthetic data、preference data、reward model 。， RM， RLHF / preference optimization。 RM ： PPO/DPO， rejection sampling、。

#### 2. Llama Nemotron：prune/distill  reasoning  agent 。

NVIDIA ： Llama ，，， post-training  RL  reasoning、instruction following、function calling  chat。Llama-Nemotron-Post-Training Dataset  math、coding、general reasoning、instruction following；OpenCodeReasoning 。Ultra  reasoning on/off，。

#### 3. RLVR  agent。

NVIDIA ，， curriculum-driven RLVR。 agent  reward 、RAG 、function calling schema、。 REINFORCE  heuristic based verifiers  instruction following / function calling ， HelpSteer2  RLHF。 RL： math/code verifier， function calling verifier； chat/RAG 。

#### 4. 。

Nemotron ， NIM、NeMo Gym、。post-training  latency、throughput、function calling 、RAG 。 AIME，。

### Mistral

> ****：[Magistral][^magistral]

Magistral  reasoning RL ：Mistral  scalable RL pipeline， RL traces， ground-up  pure RL。

#### 1. Pure RL：。

 CoT ，。Magistral  checkpoint ， RL 。 Magistral Medium  Mistral Medium 3  RL  reasoning，Magistral Small  Medium  cold-start 。： RL ； cold-start， RL。

#### 2.  reasoning language。

 reasoning RL 。Magistral  simple method to force reasoning language，，。 prompt / template  reasoning language， format reward 、。

#### 3.  RL 。

Magistral ， text data  RL  multimodal understanding、instruction following  function calling。 RL ，。，/ RL ，、、， reasoning 。

### Apple

> ****：[Apple Foundation Models 2024][^apple_fm]、[Apple Foundation Models 2025][^apple_fm_2025]

Apple  foundation model ： 3B  Apple silicon ，server model  Private Cloud Compute ，、 tool calls。

#### 1. SFT + RL ，。

2025 、、， supervised fine-tuning  reinforcement learning 。“benchmark ”， guided generation、constrained tool calling、LoRA adapter fine-tuning、。 CoT ，reward 、、、。

#### 2. ：、。

Apple  Responsible AI  locale-specific evaluation。、。 reward：/ RM ；/STEM  verifier ；tool calling schema checker 。：，。

#### 3. 。

Apple  Foundation Models framework  guided generation、constrained tool calling  LoRA。 API： constrained decoding， JSON/schema/tool ； LoRA ，。 JSON schema， constrained decoder  schema success rate。

### xAI Grok

> ****：[Grok-1][^grok_1]、[Grok 4][^grok_4]、[Grok 4.1][^grok_4_1]、[Grok 4.1 Model Card][^grok_4_1_card]

#### 1. ：， recipe 。

xAI ， post-training recipe， RL scaling、truthfulness、personality、style  emotional intelligence。“”， C 。

#### 2. ： reward。

 reward， system prompt。personality reward 、、；truthfulness reward ；emotional intelligence reward 、；safety reward 。

#### 3. ：。

 reward head  judge rubric， multi-objective DPO/RL。 reward  sycophancy，“”。

### IBM Granite

> ****：[Granite 3.3][^granite_3_3]、[Granite 4.0][^granite_4_0]、[Granite 4.1][^granite_4_1]

#### 1. ：RAG、、。

IBM Granite ：RAG、、、 thinking。Granite 3.3/4.x ，GRPO/TPO  reasoning ，。

#### 2. ：。

 Granite “”：、RAG 、 schema、/ SFT； GRPO ； RAG ； model merging  adapter merging 。

**** RAG citation faithfulness、function calling success、refusal correctness、latency、cost  thinking on/off 。

### Salesforce xLAM / SFR-RL

> ****：[Salesforce xLAM][^xlam]、[Salesforce SFR-RL][^sfr_rl]

Salesforce  xLAM / SFR-RL  agentic RL 。xLAM  action model： API ，、、。SFR-RL  agent rollout 。

#### 1. xLAM： reward 。

API agent “”，、、、schema 、。 API schema、、、。reward  schema validity、tool selection accuracy、argument exact match、execution success、final answer groundedness。 50  mock API，， verifier。

#### 2. SFR-RL：pipelined synchronous RL。

Agentic rollouts ，， on-policy 。SFR-RL  pipelined synchronous：rollout phase  training phase ， GPU 。rollout ， policy ；training ， on-policy update。 batch ， GPU ， on-policy 。

#### 3. 。

 agent rollout ， inference engine crash  batch。SFR-RL  inference gateway 、 engine actor、、 in-flight work。 scalable local-first tool execution  Expert Parallelism 。， timeout、retry、， agent RL 。

### Amazon Nova

> ****：[Amazon Nova][^nova]、[Nova Family Technical Report][^nova_report]、[Nova Premier][^nova_premier]、[Nova Forge][^nova_forge]

#### 1. Nova Forge：。

Amazon Nova ， recipe 。Nova Forge “”。 fine-tuning ， SFT；Nova Forge  pre-training、mid-training  post-training checkpoint ， Nova-curated ， RL 。

#### 2. Remote reward functions： verifier  RL。

 reward functions。 reward ，： CI，，， API 。Nova Forge  API  RL， rollout 。 Reward as a Service。

#### 3. ： verifier  reward。

 API verifier  reward： SQL、； pass/fail  rubric ； HTTP  reward。 Nova Forge ： checkpoint、 reward ，。

### Cohere Command A

> ****：[Cohere Research][^cohere_research]、[Command A][^command_a]

#### 1. Decentralized pipeline：。

Command A “”。， decentralized pipeline。 instruction following， code、safety、RAG、math、multilingual、long-context  expert track。 expert track 、。

#### 2. Expert soup：。

。 SFT Expert Models、SFT Soup Model、RL Expert Models、RL Soup Model  Polished Model。 expert track  long-context、safety、instruction、RAG & agents、multilingual、code/reasoning ；RL expert  pairwise comparisons  verifiable rewards。 polishing： RL Soup model  best-of-N supervised training， offline preference  online RL  ping-pong，。

#### 3. ：3-6  expert track + soup + polish。

Command A ： instruct ； 3-6  expert， DPO/RLVR ； model soup / task arithmetic ； polish。，、RAG、， loss 。

### Databricks, AI21, Cursor, LG, NAVER, AI2 Tulu 3

> ****：[DBRX Instruct][^dbrx]、[Jamba 1.5a][^jamba_1_5a]、[Jamba 1.5a Whitepaper][^jamba_whitepaper]、[Cursor Composer 2][^cursor_composer_2]、[EXAONE 4.0][^exaone_4_0]、[K-EXAONE][^k_exaone]、[HyperCLOVA X][^hyperclova_x]、[HyperCLOVA X THINK][^hyperclova_x_think]、[Tulu 3][^tulu_3]、[Tulu 3 Blog][^tulu_3_blog]、[RL Post-Training Survey][^rl_survey]

，。

#### 1. Databricks DBRX： instruct 。

DBRX Instruct  enterprise instruct：、、、RAG 。 eval ，。

#### 2. AI21 Jamba 1.5a：post-post-training safety alignment。

Jamba 1.5a  code of conduct 。： instruct model ，。，/， DPO  RLAIF 。

#### 3. Cursor Composer 2：coding agent 。

Cursor  coding agent ，、、、。 repo state、issue、、、 patch verifier。 GPT-5-Codex、Qwen3-Coder、MiniMax SWE Scaling 。

#### 4. LG EXAONE / NAVER HyperCLOVA X THINK： thinking 。

： benchmark，、、。thinking / non-thinking ， CoT recipe 。

#### 5. AI2 Tulu 3： post-training 。

Tulu 3 、 recipe， multi-stage post-training：SFT、preference learning、RLVR。： prompt 、、verifiable rewards、。，Tulu 3  baseline， MiniMax/Qwen/DeepSeek/Seed 。

---

## 

1. **“”“”。**  RLHF  preference pair；R1、Qwen、Seed、Mistral ；MiniMax、Kimi、LongCat、Tongyi 、。
2. **、、。** GitHub PR、Docker、Playwright、、、、。
3. **。**  cold-start SFT、reasoning RL、agentic RL、general preference / safety ； CoT 、、。
4. **。**  rollout、PD 、KV-cache 、、、reward service、LLM-as-judge  verifier，“”，。

，：

1. **。** ，，/GUI/ agent 。 reward ， RL 。
2. **。**  answer parser  verifier； checkout、、 patch ；、；GUI 、 sandbox。
3. ** SFT 。** ，、、。 RL，。
4. **。**  prompt ， verifier / judge / reward model 、、。 Qwen、DeepSeek、Kimi、MiniMax  rejection sampling / self-improvement。
5. ** RL。**  GRPO / DAPO ； value model  PPO / VAPO； agent  rollout、、 token-level credit assignment。
6. **。** 、、、， reasoning RL 。

： 5K  1K  SFT， 8 ， verifier ， GRPO ，、、。。

## 

### 

#### MiniMax

[^minimax_m2_1]: [MiniMax M2.1: Post-Training Experience and Insights for Agent Models](https://www.minimax.io/news/post-training-experience-and-insights-for-agent-models)

[^minimax_m1]: [MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585)

[^minimax_webexplorer]: [WebExplorer: Explore and Evolve for Training Long-Horizon Web Agents](https://arxiv.org/abs/2509.06501)

####  Qwen / 

[^qwen2_5]: [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)

[^qwen2_5_math]: [Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement](https://arxiv.org/abs/2409.12122)

[^qwq_32b]: [QwQ-32B: Embracing the Power of Reinforcement Learning](https://qwenlm.github.io/blog/qwq-32b/)

[^qwen3]: [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)

[^qwen3_coder]: [Qwen3-Coder: Agentic Coding in the World](https://qwenlm.github.io/blog/qwen3-coder/)

[^qwen3_coder_next]: [Qwen3-Coder-Next Technical Report](https://arxiv.org/abs/2603.00729)

[^tongyi_dr]: [Tongyi DeepResearch Technical Report](https://arxiv.org/abs/2510.24701)

#### Moonshot Kimi

[^kimi_k1_5]: [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599)

[^kimi_k2]: [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534)

[^kimi_researcher]: [Kimi-Researcher: End-to-End RL Training for Emerging Agentic Capabilities](https://moonshotai.github.io/Kimi-Researcher/)

####  Seed / Doubao

[^seed1_5_thinking]: [Seed1.5-Thinking: Advancing Superb Reasoning Models with Reinforcement Learning](https://arxiv.org/abs/2504.13914)

[^vapo]: [VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks](https://arxiv.org/abs/2504.05118)

[^dapo]: [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://seed.bytedance.com/en/public_papers/dapo-an-open-source-llm-reinforcement-learning-system-at-scale)

[^dapo_github]: [DAPO GitHub Repository](https://github.com/BytedTsinghua-SIA/DAPO)

[^seed1_5_vl]: [Seed1.5-VL Technical Report](https://arxiv.org/abs/2505.07062)

[^ui_tars]: [UI-TARS: Pioneering Automated GUI Interaction with Native Agents](https://arxiv.org/abs/2501.12326)

[^ui_tars_github]: [UI-TARS GitHub Repository](https://github.com/bytedance/ui-tars)

[^ui_tars_2]: [UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning](https://huggingface.co/papers/2509.02544)

[^seed_prover]: [Seed Prover 1.5: Advanced Mathematical Reasoning through a Novel Agentic Architecture](https://seed.bytedance.com/en/blog/seed-prover-1-5-advanced-mathematical-reasoning-through-a-novel-agentic-architecture)

[^seed1_8]: [Official Release of Seed1.8: A Generalized Agentic Model](https://seed.bytedance.com/en/blog/official-release-of-seed1-8-a-generalized-agentic-model)

#### DeepSeek

[^deepseek_math]: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)

[^deepseek_r1]: [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)

[^deepseek_v3_2]: [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556)

####  Z.ai / GLM

[^glm_4_5]: [GLM-4.5: Agentic, Reasoning, and Coding Foundation Models](https://arxiv.org/abs/2508.06471)

[^glm_5]: [GLM-5: from Vibe Coding to Agentic Engineering](https://arxiv.org/html/2602.15763v1)

####  Hunyuan

[^hunyuan_t1]: [Hunyuan-T1](https://tencent.github.io/llm.hunyuan.T1/README_EN.html)

[^hunyuan_a13b_instruct]: [Hunyuan-A13B-Instruct Model Card](https://huggingface.co/tencent/Hunyuan-A13B-Instruct)

[^hunyuan_a13b]: [Hunyuan-A13B Technical Report](https://github.com/Tencent-Hunyuan/Hunyuan-A13B/blob/main/report/Hunyuan_A13B_Technical_Report.pdf)

####  ERNIE

[^ernie_4_5_family]: [ERNIE 4.5 Model Family](https://ernie.baidu.com/blog/posts/ernie4.5/)

[^ernie_4_5]: [ERNIE 4.5 Technical Report](https://ernie.baidu.com/blog/publication/ERNIE_Technical_Report.pdf)

[^ernie_5_0]: [ERNIE 5.0 Technical Report](https://arxiv.org/abs/2602.04705)

####  StepFun

[^step3]: [Step3: Cost-Effective Multimodal Intelligence](https://stepfun.ai/research/en/step3)

[^step3_vl_10b]: [STEP3-VL-10B Technical Report](https://huggingface.co/papers/2601.09668)

[^step_deepresearch]: [Step-DeepResearch Technical Report](https://arxiv.org/abs/2512.20491)

####  LongCat

[^longcat_flash]: [LongCat-Flash-Thinking-2601 ](https://tech.meituan.com/2026/02/02/longcat-flash-thinking-2601-techreport.html)

####  Ling / Ring

[^ling_1t]: [Ling-1T Model](https://ant-ling.medium.com/deep-insight-efficient-inference-introducing-the-trillion-parameter-ling-1t-model-77d6170e5e8e)

[^ring_1t]: [Ring-1T](https://ant-ling.medium.com/ring-1t-release-the-flow-state-of-insight-born-of-epiphany-c20e8e32817c)

####  Pangu

[^pangu_ultra]: [Pangu Ultra](https://github.com/pangu-tech/pangu-ultra)

[^pangu_pro_moe]: [Pangu Pro MoE: Mixture of Grouped Experts for Efficient Sparsity](https://arxiv.org/abs/2505.21411)

[^pangu_news]: [ 7B  72B ](https://www.huawei.com/cn/news/2025/7/pangu-opensource)

#### 01.AI Yi

[^yi_lightning]: [Yi-Lightning Technical Report](https://arxiv.org/abs/2412.01253)

#### InternLM /  AI Lab

[^internlm2]: [InternLM2 Technical Report](https://arxiv.org/abs/2403.17297)

####  Baichuan  360 

[^baichuan2]: [Baichuan 2: Open Large-scale Language Models](https://arxiv.org/abs/2309.10305)

[^zhinao]: [360Zhinao Technical Report](https://arxiv.org/abs/2405.13386)

####  Skywork  MiMo

[^skywork_or1]: [Skywork Open Reasoner 1 Technical Report](https://huggingface.co/papers/2505.22312)

[^skywork_or1_github]: [Skywork-OR1 GitHub Repository](https://github.com/SkyworkAI/Skywork-OR1)

[^mimo]: [MiMo: Unlocking the Reasoning Potential of Language Model -- From Pretraining to Posttraining](https://arxiv.org/abs/2505.07608)

[^mimo_github]: [Xiaomi MiMo GitHub Repository](https://github.com/XiaomiMiMo/MiMo)

[^mimo_vl]: [Xiaomi MiMo-VL-Miloco Technical Report](https://arxiv.org/abs/2512.17436)

#### 、、

[^keye_vl]: [Kwai Keye-VL Technical Report](https://arxiv.org/abs/2507.01949)

[^sensenova_u1]: [SenseNova U1](https://www.sensetime.com/en/news-detail/51170629?categoryId=1072)

[^spark_x1]: [Spark X1 deep reasoning model](https://news.cgtn.com/news/2025-01-15/China-releases-Spark-X1-deep-reasoning-model-that-packs-a-punch-1AbIq8PzzEI/index.html)

### 

#### OpenAI

[^instructgpt]: [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

[^gpt4]: [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)

[^o1]: [OpenAI o1 System Card](https://openai.com/index/openai-o1-system-card/)

[^o3_o4_mini]: [OpenAI o3 and o4-mini System Card](https://openai.com/index/o3-o4-mini-system-card/)

[^o3_operator]: [Addendum to o3 and o4-mini system card: OpenAI o3 Operator](https://openai.com/index/o3-o4-mini-system-card-addendum-operator-o3/)

[^gpt4_5]: [OpenAI GPT-4.5 System Card](https://openai.com/index/gpt-4-5-system-card/)

[^gpt5]: [OpenAI GPT-5 System Card](https://openai.com/index/gpt-5-system-card/)

[^gpt5_1]: [Addendum to GPT-5 system card: GPT-5.1](https://openai.com/index/gpt-5-system-card-addendum-gpt-5-1/)

[^gpt5_4]: [OpenAI GPT-5.4 Thinking System Card](https://openai.com/index/gpt-5-4-thinking-system-card/)

[^gpt5_5]: [OpenAI GPT-5.5 System Card](https://openai.com/index/gpt-5-5-system-card/)

[^gpt5_5_instant]: [OpenAI GPT-5.5 Instant System Card](https://openai.com/index/gpt-5-5-instant-system-card/)

[^gpt5_codex]: [Addendum to GPT-5 system card: GPT-5-Codex](https://openai.com/index/gpt-5-system-card-addendum-gpt-5-codex/)

[^gpt5_1_codex_max]: [GPT-5.1-Codex-Max System Card](https://openai.com/index/gpt-5-1-codex-max-system-card/)

[^gpt5_2_codex]: [Introducing GPT-5.2-Codex](https://openai.com/index/introducing-gpt-5-2-codex/)

#### Anthropic

[^constitutional_ai]: [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)

[^anthropic_cai]: [Anthropic Constitutional AI overview](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback)

[^claude4]: [System Card: Claude Opus 4 & Claude Sonnet 4](https://www.anthropic.com/claude-4-system-card)

[^claude_sonnet_4_5]: [Claude Sonnet 4.5 System Card](https://www.anthropic.com/claude-sonnet-4-5-system-card)

[^claude_opus_4_5]: [Claude Opus 4.5 System Card](https://www.anthropic.com/claude-opus-4-5-system-card)

[^claude_opus_4_6]: [Claude Opus 4.6 System Card](https://www-cdn.anthropic.com/0dd865075ad3132672ee0ab40b05a53f14cf5288.pdf)

#### Google DeepMind

[^gemini_1_5]: [Gemini 1.5 Technical Report](https://arxiv.org/abs/2403.05530)

[^gemini_2_5]: [Gemini 2.5 Technical Report](https://arxiv.org/abs/2507.06261)

[^gemini_2_5_deep_think]: [Gemini 2.5 Deep Think](https://blog.google/products/gemini/gemini-2-5-deep-think)

[^gemini_2_5_computer_use]: [Gemini 2.5 Computer Use Model](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-computer-use-model/)

[^gemini_3_1_pro]: [Gemini 3.1 Pro Model Card](https://deepmind.google/models/model-cards/gemini-3-1-pro/)

[^gemma_3]: [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786)

#### Meta Llama

[^llama3_herd]: [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)

#### Microsoft Phi

[^phi_4]: [Phi-4 Technical Report](https://arxiv.org/abs/2412.08905)

[^phi_4_reasoning]: [Phi-4-reasoning Technical Report](https://arxiv.org/abs/2504.21318)

#### NVIDIA Nemotron

[^nemotron_4]: [Nemotron-4 340B Technical Report](https://arxiv.org/abs/2406.11704)

[^llama_nemotron]: [Llama-Nemotron: Efficient Reasoning Models](https://arxiv.org/abs/2505.00949)

[^nemotron_ultra]: [NVIDIA Llama Nemotron Ultra Open Model](https://developer.nvidia.com/blog/nvidia-llama-nemotron-ultra-open-model-delivers-groundbreaking-reasoning-accuracy/)

[^nemotron_agents]: [Build Enterprise AI Agents with NVIDIA Llama Nemotron Reasoning Models](https://developer.nvidia.com/blog/build-enterprise-ai-agents-with-advanced-open-nvidia-llama-nemotron-reasoning-models/)

[^nemotron_h]: [Nemotron-H Reasoning Model Family](https://developer.nvidia.com/blog/nemotron-h-reasoning-enabling-throughput-gains-with-no-compromises/)

[^nemotron_3]: [Inside NVIDIA Nemotron 3](https://developer.nvidia.com/blog/inside-nvidia-nemotron-3-techniques-tools-and-data-that-make-it-efficient-and-accurate/)

#### Mistral

[^magistral]: [Magistral](https://arxiv.org/abs/2506.10910)

#### Apple

[^apple_fm]: [Apple Intelligence Foundation Language Models](https://machinelearning.apple.com/research/apple-intelligence-foundation-language-models)

[^apple_fm_2025]: [Apple Intelligence Foundation Language Models Tech Report 2025](https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025)

#### xAI Grok

[^grok_1]: [xAI Grok-1 Model Card](https://x.ai/news/grok/model-card)

[^grok_4]: [xAI Grok 4](https://x.ai/news/grok-4)

[^grok_4_1]: [xAI Grok 4.1](https://x.ai/news/grok-4-1/)

[^grok_4_1_card]: [xAI Grok 4.1 Model Card](https://data.x.ai/2025-11-17-grok-4-1-model-card.pdf)

#### IBM Granite

[^granite_3_3]: [IBM Granite 3.3](https://www.ibm.com/new/announcements/ibm-granite-3-3-speech-recognition-refined-reasoning-rag-loras)

[^granite_4_0]: [IBM Granite 4.0](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models)

[^granite_4_1]: [IBM Granite 4.1 Build Notes](https://huggingface.co/blog/ibm-granite/granite-4-1)

#### Salesforce xLAM / SFR-RL

[^xlam]: [Salesforce xLAM](https://www.salesforce.com/blog/large-action-model-ai-agent/)

[^sfr_rl]: [Salesforce SFR-RL](https://www.salesforce.com/blog/efficient-rl-training-agentic-era/)

#### Amazon Nova

[^nova]: [Amazon Nova](https://aws.amazon.com/nova/)

[^nova_report]: [The Amazon Nova Family of Models: Technical Report and Model Card](https://www.isi.edu/results/publications/31887/the-amazon-nova-family-of-models-technical-report-and-model-card/)

[^nova_premier]: [Amazon Nova Premier: Technical report and model card](https://www.amazon.science/publications/amazon-nova-premier-technical-report-and-model-card)

[^nova_forge]: [Amazon Nova Forge](https://aws.amazon.com/nova/forge/)

#### Cohere Command A

[^cohere_research]: [Cohere Research](https://cohere.com/research)

[^command_a]: [Command A: An Enterprise-Ready Large Language Model](https://cohere.com/research/papers/command-a-technical-report.pdf)

#### Databricks

[^dbrx]: [DBRX Instruct](https://huggingface.co/databricks/dbrx-instruct)

#### AI21

[^jamba_1_5a]: [Jamba 1.5a: Enhancing AI Safety Through Post-Post-Training Alignment](https://www.ai21.com/research/jamba-1-5a/)

[^jamba_whitepaper]: [Jamba 1.5a Whitepaper](https://lp.ai21.com/hubfs/resources/Jamba-1-5a-Whitepaper.pdf)

#### Cursor

[^cursor_composer_2]: [Cursor Composer 2 Technical Report](https://cursor.com/blog/composer-2-technical-report)

#### LG EXAONE

[^exaone_4_0]: [EXAONE 4.0 Technical Report](https://www.lgresearch.ai/data/cdn/upload/EXAONE_4_0.pdf)

[^k_exaone]: [K-EXAONE Technical Report](https://www.lgresearch.ai/data/cdn/upload/K-EXAONE_Technical_Report.pdf)

#### NAVER HyperCLOVA X

[^hyperclova_x]: [HyperCLOVA X Technical Report](https://arxiv.org/abs/2404.01954)

[^hyperclova_x_think]: [HyperCLOVA X THINK Technical Report](https://huggingface.co/papers/2506.22403)

### 

#### AI2 Tulu / Survey

[^tulu_3]: [Tulu 3: Pushing Frontiers in Open Language Model Post-Training](https://openreview.net/forum?id=i1uGbfHHpH)

[^tulu_3_blog]: [Tulu 3 Technical Blog](https://allenai.org/blog/tulu-3-technical)

[^rl_survey]: [Reinforcement Learning for LLM Post-Training: A Survey](https://openreview.net/forum?id=UdsXTNzzvg)
