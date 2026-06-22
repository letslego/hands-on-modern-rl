"""
9：GRPO  —— GSM8K 
==========================================================

 GRPO （Qwen2.5-0.5B-Instruct）
， RLVR (Reinforcement Learning with Verifiable Rewards) 。

：
  1.  GSM8K （20 ）
  2.  Qwen2.5-0.5B-Instruct 
  3.  group_size=4 
  4. 
  5. GRPO 
  6. （）
  7.  3  epoch，、、

：
  pip install -r requirements.txt
  python grpo_math_reasoning.py

：
  -  CPU ，
  -  GPU（ 2~4 GB ）
  -  GPU，
"""

import re
import copy
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


# ==========================================
# ：GSM8K 
# ==========================================
# 20 ，
#  question（） answer（）

math_dataset = [
    {
        "question": " 15 ， 27 ，？",
        "answer": "42",
    },
    {
        "question": " 36 ， 6 ，？",
        "answer": "6",
    },
    {
        "question": " 5 ，（7）？",
        "answer": "35",
    },
    {
        "question": " 240 ， 85 ，？",
        "answer": "155",
    },
    {
        "question": " 48 ， 36 ， 25 ，？",
        "answer": "109",
    },
    {
        "question": " 24 ，8 ？",
        "answer": "192",
    },
    {
        "question": " 100 ， 32 ， 8 ，？",
        "answer": "60",
    },
    {
        "question": " 12 ， 8 ，？",
        "answer": "96",
    },
    {
        "question": " 5 ， 12 ，？",
        "answer": "60",
    },
    {
        "question": " 15 ， 8 ，？",
        "answer": "120",
    },
    {
        "question": " 120 ， 3 ，？",
        "answer": "360",
    },
    {
        "question": " 45 ， 18 ，？",
        "answer": "63",
    },
    {
        "question": " 10 ， 6 ，？",
        "answer": "30",
    },
    {
        "question": " 3 ， 8 ， 2 ， 5 ，？",
        "answer": "34",
    },
    {
        "question": " 280 ， 95 ，？",
        "answer": "185",
    },
    {
        "question": " 15 ， 60 ，？",
        "answer": "900",
    },
    {
        "question": " 50 ， 25 ，？",
        "answer": "150",
    },
    {
        "question": " 4  72 ，？",
        "answer": "18",
    },
    {
        "question": " 500 ， 468 ，？",
        "answer": "32",
    },
    {
        "question": " 18 ，，？",
        "answer": "12",
    },
]


# ==========================================
# ：
# ==========================================
def extract_answer_from_response(response):
    """
    

    ：
      1. \\boxed{...} 
      2. "/：" 
      3. （）

    ：
        response: 
    ：
        str  None: 
    """
    # 1：\boxed{} 
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
    if boxed_match:
        return boxed_match.group(1).strip()

    # 2：
    cn_match = re.search(r'[：:]\s*([+-]?\d+\.?\d*)', response)
    if cn_match:
        return cn_match.group(1).strip()

    # 3：
    en_match = re.search(r'[Tt]he answer is\s*([+-]?\d+\.?\d*)', response)
    if en_match:
        return en_match.group(1).strip()

    # 4：（）
    numbers = re.findall(r'([+-]?\d+\.?\d*)', response)
    if numbers:
        return numbers[-1]

    return None


def compute_reward(response, ground_truth):
    """
    

    ，：
      - ： = 1.0
      - ： = 0.0
      - ：， +0.1
      - （）， +0.1

     1.0

    ：
        response: 
        ground_truth: 
    ：
        float:  [0.0, 1.0]
    """
    reward = 0.0

    # 
    extracted = extract_answer_from_response(response)
    if extracted is not None:
        try:
            if abs(float(extracted) - float(ground_truth)) < 1e-6:
                reward = 1.0
        except (ValueError, TypeError):
            if str(extracted).strip() == str(ground_truth).strip():
                reward = 1.0

    # （，）
    if reward > 0:
        has_steps = bool(re.search(r'|\d+|Step||||', response))
        if has_steps:
            reward = min(1.0, reward + 0.1)

    return reward


# ==========================================
# ：GRPO 
# ==========================================
def compute_grpo_advantages(rewards):
    """
    GRPO ：

    ：
        advantage_i = (reward_i - mean) / (std + eps)

    ：
        rewards: list[float]，
    ：
        numpy ：
    """
    rewards_arr = np.array(rewards, dtype=np.float64)
    mean_r = rewards_arr.mean()
    std_r = rewards_arr.std() + 1e-8
    advantages = (rewards_arr - mean_r) / std_r
    return advantages


# ==========================================
# ：
# ==========================================
def generate_responses(model, tokenizer, prompt, num_responses=4, max_new_tokens=200):
    """
    

     do_sample=True + temperature > 0 
    ，

    ：
        model: 
        tokenizer: 
        prompt: 
        num_responses: （group_size）
        max_new_tokens:  token 
    ：
        list[str]: 
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    responses = []
    for _ in range(num_responses):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        # ：（ prompt）
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True,
        )
        responses.append(response)

    return responses


# ==========================================
# ：GRPO 
# ==========================================
def grpo_policy_update(model, tokenizer, optimizer, prompt, responses, advantages):
    """
     GRPO 

    ：
        loss = -mean(advantage_i * log_prob_i)

     log_prob_i  i 。
    ， token 。

    ：
        model: 
        tokenizer: 
        optimizer: 
        prompt: 
        responses: 
        advantages: GRPO 
    ：
        float: 
    """
    model.train()

    #  prompt  token ids
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = prompt_ids.shape[-1]

    total_loss = 0.0
    num_valid = 0

    for response, advantage in zip(responses, advantages):
        if advantage <= 0:
            continue  # ，

        #  prompt + response 
        response_ids = tokenizer.encode(response, return_tensors="pt").to(model.device)
        #  response_ids  BOS token
        if response_ids[0, 0] == tokenizer.bos_token_id:
            response_ids = response_ids[:, 1:]

        full_ids = torch.cat([prompt_ids, response_ids], dim=-1)

        #  logits
        with torch.cuda.amp.autocast(enabled=False):
            outputs = model(full_ids)
            logits = outputs.logits

        #  log_prob（ response  token）
        # logits[:, :-1, :]  t ，full_ids[:, 1:]  t+1  token
        response_logits = logits[0, prompt_len - 1:-1, :]  # response  logits
        response_tokens = full_ids[0, prompt_len:]          # response  token ids

        if response_tokens.shape[0] == 0:
            continue

        #  token 
        log_probs = torch.nn.functional.log_softmax(response_logits, dim=-1)
        token_log_probs = log_probs.gather(1, response_tokens.unsqueeze(1)).squeeze(1)

        # （ token ，）
        avg_log_prob = token_log_probs.mean()

        # ：-advantage * log_prob
        loss = -advantage * avg_log_prob
        total_loss += loss
        num_valid += 1

    if num_valid == 0:
        return 0.0

    # 
    avg_loss = total_loss / num_valid

    # 
    optimizer.zero_grad()
    avg_loss.backward()
    # ，
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return avg_loss.item()


# ==========================================
# ：
# ==========================================
def train():
    """
    GRPO 

    ：
        - model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        - group_size = 4： 4 
        - num_epochs = 3：
        - learning_rate = 1e-5：
        - max_new_tokens = 200： token 
    """
    # ----------  ----------
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    group_size = 4       # （GRPO  group）
    num_epochs = 3       # 
    learning_rate = 1e-5 # 
    max_new_tokens = 200 # 

    print("=" * 70)
    print("  GRPO （GSM8K ）")
    print("=" * 70)
    print(f"  : {model_name}")
    print(f"  : {len(math_dataset)} ")
    print(f"   (group_size): {group_size}")
    print(f"  : {num_epochs}")
    print(f"  : {learning_rate}")
    print(f"  : {max_new_tokens}")
    print("=" * 70)

    # ----------  ----------
    print("\n...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  : {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device,
    )
    model.eval()  # 

    # 
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # ----------  ----------
    print("\n" + "=" * 70)
    print("  （ 3 ）")
    print("=" * 70)
    test_indices = [0, 7, 14]  #  1、8、15 
    pre_train_results = evaluate_on_samples(model, tokenizer, math_dataset, test_indices)

    # ----------  ----------
    print("\n" + "=" * 70)
    print("   GRPO ")
    print("=" * 70)

    # 
    epoch_metrics = []

    for epoch in range(num_epochs):
        print(f"\n{'#' * 70}")
        print(f"  Epoch {epoch + 1}/{num_epochs}")
        print(f"{'#' * 70}")

        epoch_total_rewards = []
        epoch_accuracies = []
        epoch_lengths = []
        epoch_losses = []

        for idx, problem in enumerate(math_dataset):
            question = problem["question"]
            ground_truth = problem["answer"]

            # Step 1：
            responses = generate_responses(
                model, tokenizer, question,
                num_responses=group_size,
                max_new_tokens=max_new_tokens,
            )

            # Step 2：
            rewards = []
            for resp in responses:
                r = compute_reward(resp, ground_truth)
                rewards.append(r)

            # Step 3：GRPO 
            advantages = compute_grpo_advantages(rewards)

            # Step 4：
            loss = grpo_policy_update(
                model, tokenizer, optimizer,
                question, responses, advantages,
            )

            # 
            avg_reward = np.mean(rewards)
            accuracy = np.mean([1.0 if r >= 1.0 else 0.0 for r in rewards])
            avg_length = np.mean([len(r) for r in responses])

            epoch_total_rewards.append(avg_reward)
            epoch_accuracies.append(accuracy)
            epoch_lengths.append(avg_length)
            epoch_losses.append(loss)

            #  5 
            if (idx + 1) % 5 == 0:
                recent_acc = np.mean(epoch_accuracies[-5:])
                recent_reward = np.mean(epoch_total_rewards[-5:])
                print(
                    f"   {idx + 1:2d}/{len(math_dataset)} | "
                    f"5: {recent_acc:.1%} | "
                    f": {recent_reward:.3f} | "
                    f": {loss:.4f}"
                )

        # Epoch ，
        epoch_metrics.append({
            "epoch": epoch + 1,
            "avg_reward": np.mean(epoch_total_rewards),
            "accuracy": np.mean(epoch_accuracies),
            "avg_length": np.mean(epoch_lengths),
            "avg_loss": np.mean([l for l in epoch_losses if l != 0.0]),
        })

        print(f"\n  Epoch {epoch + 1} :")
        print(f"    :   {epoch_metrics[-1]['avg_reward']:.4f}")
        print(f"    : {epoch_metrics[-1]['accuracy']:.2%}")
        print(f"    : {epoch_metrics[-1]['avg_length']:.0f} ")
        print(f"    :   {epoch_metrics[-1]['avg_loss']:.4f}")

    # ----------  ----------
    print("\n" + "=" * 70)
    print("  （ 3 ）")
    print("=" * 70)
    post_train_results = evaluate_on_samples(model, tokenizer, math_dataset, test_indices)

    # ----------  ----------
    print("\n" + "=" * 70)
    print("  ")
    print("=" * 70)
    for i, test_idx in enumerate(test_indices):
        problem = math_dataset[test_idx]
        pre = pre_train_results[i]
        post = post_train_results[i]
        print(f"\n  : {problem['question'][:40]}...")
        print(f"  : {problem['answer']}")
        print(f"  : {pre['response'][:60]}... → : {pre['reward']:.2f}")
        print(f"  : {post['response'][:60]}... → : {post['reward']:.2f}")

    # ----------  ----------
    print("\n" + "=" * 70)
    print("  ")
    print("=" * 70)
    print()
    print(f"  {'Epoch':>5s}  {'':>10s}  {'':>10s}  {'':>10s}  {'':>10s}")
    print(f"  {'─────':>5s}  {'──────────':>10s}  {'──────────':>10s}  {'──────────':>10s}  {'──────────':>10s}")
    for m in epoch_metrics:
        print(f"  {m['epoch']:>5d}  {m['avg_reward']:>10.4f}  "
              f"{m['accuracy']:>10.2%}  {m['avg_length']:>10.0f}  "
              f"{m['avg_loss']:>10.4f}")

    # ----------  ----------
    print("\n" + "=" * 70)
    print("  GRPO ")
    print("=" * 70)
    print(f"""
   GRPO + RLVR ：

  1. ：
     - {len(math_dataset)}  GSM8K 
     - ，

  2. GRPO ：
     -  {group_size} 
     - 
     - ， Critic 
     - 

  3. ：
     - ：，
     - ：
     - 

  4.  DPO ：
     - DPO （chosen/rejected）
     - GRPO 
     - （、），GRPO + RLVR 

  5. ：
     - （ GSM8K ）
     - （Qwen2.5-1.5B / 7B）
     -  group_size（8~16）
     -  DeepSeek-R1 
    """)


# ==========================================
# ：
# ==========================================
def evaluate_on_samples(model, tokenizer, dataset, indices):
    """
    （）

    ：
        model: 
        tokenizer: 
        dataset: 
        indices: 
    ：
        list[dict]: 
    """
    model.eval()
    results = []

    for idx in indices:
        problem = dataset[idx]
        # （）
        messages = [{"role": "user", "content": problem["question"]}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
            )

        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True,
        )

        reward = compute_reward(response, problem["answer"])
        extracted = extract_answer_from_response(response)

        result = {
            "question": problem["question"],
            "ground_truth": problem["answer"],
            "response": response,
            "extracted_answer": extracted,
            "reward": reward,
            "correct": str(extracted) == str(problem["answer"]) if extracted else False,
        }
        results.append(result)

        # 
        print(f"\n  : {problem['question'][:50]}...")
        print(f"  : {problem['answer']}")
        print(f"  : {response[:80]}{'...' if len(response) > 80 else ''}")
        print(f"  : {extracted}")
        print(f"  : {'' if result['correct'] else ''} | : {reward:.2f}")

    return results


# ==========================================
# 
# ==========================================
if __name__ == "__main__":
    train()
