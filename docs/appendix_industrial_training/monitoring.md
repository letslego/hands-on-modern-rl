---
search: false
---

# ：（ B.3）

> 。 [B.3 RL  Agentic RL Benchmark](./evaluation-badcase) “”。，。

> RL 。reward ，，KL ——。
>
> ：**，？**

## 

 6 ：

|                 |    |                         |
| ------------------- | ---------- | ------------------------------------- |
| **Training Reward** |    |  =                    |
| **KL Divergence**   |    |  =                    |
| **Policy Entropy**  |    |  = （） |
| **Clip Fraction**   | < 0.3      |  > 0.3 =                |
| **Value Loss**      |    |  = Critic                   |
| **Reward Margin**   |  |  = Reward Model               |

****：Reward  Entropy —— reward hacking（""）， [ A](/appendix_common_pitfalls/intro)。

## 

|                   |        |                            |
| --------------------- | -------------- | ---------------------------------- |
| Reward            |        | ， clip epsilon      |
| KL                |    |  batch size  KL  |
| Entropy           |        |  entropy bonus                 |
|           | Reward hacking |  RM      |
| OOM                   | /  |  FSDP/ZeRO-3， batch size  |
| Reward  | RM         |  checkpoint， RM |

## 

- **Weights & Biases (wandb)**：，，
- **TensorBoard**：PyTorch ，
- ****：Grafana + Prometheus，

```python
# wandb 
import wandb
wandb.init(project="my-rl-training")
wandb.log({
    "reward": mean_reward,
    "kl_divergence": kl_div,
    "entropy": entropy,
    "clip_fraction": clip_frac,
})
```

## 

。：

1. **KL **（ 0.1-0.15）——
2. **Reward  N **（N ， 50-100 ）
3. **Entropy  0**——，
4. ** benchmark  2 **——

： 100-500  benchmark，——， checkpoint。 [B.3 RL  Agentic RL Benchmark](./evaluation-badcase) 。
