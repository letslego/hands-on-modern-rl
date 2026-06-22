# 8.7 veRL + GSM8K 

 [8.7 ： veRL  GSM8K  PPO ](../../../docs/chapter08_rlhf/verl-ppo-gsm8k.md)。

 veRL 。 GSM8K reward ； veRL 。

## 

- veRL ：<https://github.com/volcengine/verl>
- veRL PPO ：`python3 -m verl.trainer.main_ppo`
- veRL GSM8K ：`examples/data_preprocess/gsm8k.py`

## 

 veRL， GSM8K ：

```bash
git clone https://github.com/volcengine/verl.git
cd verl
pip install -e .
python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k
```

 veRL ：

```bash
cd /path/to/hands-on-modern-rl/code/chapter08_rlhf/verl_gsm8k
chmod +x run_qwen2_5_0_5b_ppo_single_gpu.sh
./run_qwen2_5_0_5b_ppo_single_gpu.sh
```

 reward：

```bash
./run_qwen2_5_0_5b_ppo_single_gpu.sh \
  custom_reward_function.path="$(pwd)/gsm8k_reward_advanced.py" \
  custom_reward_function.name=compute_score
```

## 

|                                  |                             |
| ------------------------------------ | ------------------------------- |
| `gsm8k_reward.py`                    |  0/1 accuracy reward        |
| `gsm8k_reward_advanced.py`           | accuracy + format  reward |
| `run_qwen2_5_0_5b_ppo_single_gpu.sh` |  0.5B PPO           |
| `run_qwen2_5_0_5b_ppo_8gpu.sh`       |  8  PPO           |
