# 11.3 VLM RL ——

 VLM GRPO ， VLM RL 。""——、、。""——， VLM RL 。

## 11.3.1 VisPlay：

VisPlay  VLM RL ， RL **、**——""（Questioner），""（Reasoner）。 8  Self-Play ，。

![VisPlay Framework](./images/ref-visplay-framework.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em> 1：VisPlay 。Image-Conditioned Questioner ，Multimodal Reasoner ，、 GRPO 。：<a href="https://bruno686.github.io/VisPlay/" target="_blank" rel="noopener noreferrer">VisPlay Project Page</a></em>
</div>

### 

**（Questioner）** 。，。：（），（）。

**（Reasoner）** 。，。 VLM GRPO ， RL 。

： →  → "" → …… AlphaGo （ 5 ）——。

VisPlay 。——，，；，，。：（），。""——。

——、、。：（）。，VLM ，。

## 11.3.2 VISTA-Gym： RL 

VISTA-Gym " VLM ，"。 Python 、、 VLM ——，。

![VISTA-Gym Overview](./images/ref-vista-gym-overview.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em> 2：VISTA-Gym 、、VLM Agent、。，、。：<a href="https://www.eigenai.com/blog/vista-gym-vista-r1" target="_blank" rel="noopener noreferrer">VISTA-Gym / VISTA-R1 Blog</a></em>
</div>

### 

：，"？"。 VLM ——""。，：

1. ****：""
2. ****：
3. ****：" + "
4. ****："，，"

VISTA-Gym """"。 10 ， 2 。：

$$R_{total} = R_{accuracy} + R_{reasoning} - \lambda \cdot N_{tools}$$

 $N_{tools}$ ，$\lambda$ 。

###  GRPO 

VISTA-Gym  GRPO 。+，（），，，。 9  GRPO ——""" + "。

## 11.3.3 

 VisPlay、VISTA-Gym  VLM GRPO ：

![VISTA-R1 Results](./images/ref-vista-gym-results.png)

<div style="text-align: center; font-size: 0.9em; color: var(--vp-c-text-2); margin-top: -10px; margin-bottom: 20px;">
  <em> 3：VISTA-R1 。、RL 、， VLM RL “ +  + reward + ”。：<a href="https://www.eigenai.com/blog/vista-gym-vista-r1" target="_blank" rel="noopener noreferrer">VISTA-Gym / VISTA-R1 Blog</a></em>
</div>

|              | VLM GRPO（）           | VisPlay                    | VISTA-Gym               |
| ------------ | ---------------------------- | -------------------------- | ----------------------- |
| **** |  GRPO  VLM     | +      |         |
| **** |          |      |  +      |
| **** | （++） |        |  +  |
| ****     | ，           | ，     | ，      |
| ****     | ，         | ， |       |
| **** | 、           |                |       |

，。VLM GRPO ""——。VisPlay ""——。VISTA-Gym ""——。，： GRPO ， VisPlay ， VISTA-Gym 。

## 11.3.4 VLM RL 

VLM RL ，：

### 

，。（），（）。VLM RL ：（），（ 1 ），（""""）。

### 3D 

 2D  3D ，VLM 、。。3D ****——。RL 。

###  VLM-RL

VLM-RL 。，。 12.1 ，VLM-RL ""——，""，""。

|           |                  |                        |                        |
| ------------- | -------------------------------- | -------------------------- | -------------------------- |
|  RL |            | ，           | ，     |
| VLM-RL        | ，       | 、 | 、 |
|       | VLM ， |              |      |

" → "：****——（"，"）；****——""，""；****——，VLM 。

###  VLM-RL 

 VLM-RL " →  → "：

****， RL  VLM 。（），。

****，。（Domain Randomization， 12.1 ）——、、，。

****，，。——（、、）。

```python
# ==========================================
#  VLM-RL 
# ==========================================
def robot_vlm_rl_train(vlm, simulator, num_episodes=10000):
    """ VLM-RL """
    optimizer = setup_optimizer_with_lr_decay(vlm)
    best_reward = -float('inf')

    for episode in range(num_episodes):
        # 1. 
        scene = simulator.reset()
        image = simulator.render_camera()  # 

        # 2. VLM 
        scene_desc = vlm.describe(image)
        action_plan = vlm.plan_action(scene_desc)

        # 3. 
        total_reward = 0
        for action in action_plan:
            obs, reward, done, info = simulator.step(action)
            total_reward += reward

            # （）
            if info.get('collision', False):
                total_reward -= 10.0
                break

        # 4.  GRPO  PPO 
        loss = compute_policy_gradient_loss(vlm, episode_data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vlm.parameters(), max_norm=1.0)
        optimizer.step()

        # 5. 
        if total_reward > best_reward:
            best_reward = total_reward
            save_model(vlm, 'best_vlm_robot.pt')

        if (episode + 1) % 1000 == 0:
            eval_reward = evaluate(vlm, simulator, num_episodes=50)
            print(f"Episode {episode+1} | "
                  f": {total_reward:.1f} | "
                  f": {eval_reward:.1f}")
```

## 11.3.5  RL  RL：

， RL ：

|                  |             |          |                 |   |
| -------------------- | --------------- | ---------------- | ----------------------- | --------- |
|  4 ：DQN         | /   |              |                 | DQN       |
|  5 ：    |         | /        |                 | REINFORCE |
|  8 ：RLHF/PPO    | Token       | （token）    | RM                  | PPO       |
|  9 ：GRPO        | Token       | （token）    |                 | GRPO      |
|  10 ：Agentic RL |  +  |  / token |  +      | PPO/GRPO  |
|  11 ：VLM RL     |  + Token    | （token）    |  +  + grounding | GRPO      |

——（ 5 ）、Actor-Critic （ 6 ）、PPO （ 7 ）、GRPO （ 9 ）。、。——。

<details>
<summary>： VLM ，GRPO ？</summary>

 GRPO （、）。： ViT （ TimeSformer  ViViT）。 token ""，""。

——""，。""，，""""。

</details>

VLM RL 。 GPT-4V  Gemini， LLaVA  Qwen-VL， RL 。——、、、……。

## 11.3.6  VLM RL  Agent

VLM RL ""。，"、"—— + （GUI Agent）、 + （Data Agent）。 VLM RL  Agent ：** + **。

###  Agent 

|              |       |            |  Agent  |
| ---------------- | --------- | -------------------- | ------------------- |
|      | 📊    | 、   | ❌        |
|  Bug   | 📸    | 、     | ❌  UI        |
|      | 🖼️  | 、 API     | ❌      |
|  | 🏥 CT/MRI | 、 | ❌      |

###  Agent RL 

 VLM RL [ 10 ](../chapter10_agentic_rl/intro) Agent RL ，：

**1. 。**  Agent ，（""）（""）。—— VLM RL （）， [Agent RL ](../chapter10_agentic_rl/tool-use-and-trajectory)。****：，。

**2. 。**  Agent  reward ， Agent  reward ：

```python
def multimodal_agent_reward(trajectory, task):
    """ Agent """
    visual_reward = evaluate_visual_understanding(task.image, trajectory.visual_description)
    tool_reward = evaluate_tool_usage(task.required_tools, trajectory.tool_calls)
    outcome_reward = task.verify_final_result(trajectory.final_output)
    return 0.2 * visual_reward + 0.3 * tool_reward + 0.5 * outcome_reward
```

**3.  Credit Assignment。**  10 ， 2  5 。 Agent  credit assignment ，、。[ 10 ](../chapter10_agentic_rl/multi-turn-rl) ORM vs PRM 。

### 

**GUI Agent。**  RL  UI （、），、、。 CRAFT-GUI（ GUI ）、MobileRL（）。GUI Agent  RLVR ——。

** Deep Research。** [Tongyi DeepResearch](../chapter10_agentic_rl/deep-research-agent) ，、 PDF 。 VLM RL + Agent RL 。

** Agent。** ，/。 reward ——""， LLM-as-Judge 。

### 

 Agent，：

1. ****： VLM GRPO 
2. ****：[ 10  RL](../chapter10_agentic_rl/tool-use-and-trajectory)
3. ****： Agent  RL，reward 

：**，**。，。

""""—— Diffusion  RL 、。

## 

- [VisPlay Project Page](https://bruno686.github.io/VisPlay/) ——  Image-Conditioned Questioner  Multimodal Reasoner 。
- [VISTA-Gym / VISTA-R1 Blog](https://www.eigenai.com/blog/vista-gym-vista-r1) —— 、VISTA-R1 。
