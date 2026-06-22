import torch
import torch.nn.functional as F


# [A] ： prompt  chosen  rejected
example = {
    "prompt": "：，？",
    "chosen": "，。",
    "rejected": "，，。",
}


# [B]  log probability
def sequence_logprob(model, input_ids, attention_mask, labels):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    target_ids = input_ids[:, 1:]
    target_mask = labels[:, 1:].ne(-100)

    token_logprobs = logits.log_softmax(dim=-1)
    picked_logprobs = token_logprobs.gather(
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)

    answer_logprobs = picked_logprobs * target_mask
    return answer_logprobs.sum(dim=-1)


# [C] Policy ，Reference ，
def dpo_forward(policy_model, ref_model, batch, beta=0.1):
    chosen_logps = sequence_logprob(
        policy_model,
        batch["chosen_input_ids"],
        batch["chosen_attention_mask"],
        batch["chosen_labels"],
    )
    rejected_logps = sequence_logprob(
        policy_model,
        batch["rejected_input_ids"],
        batch["rejected_attention_mask"],
        batch["rejected_labels"],
    )

    with torch.no_grad():
        ref_chosen_logps = sequence_logprob(
            ref_model,
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["chosen_labels"],
        )
        ref_rejected_logps = sequence_logprob(
            ref_model,
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch["rejected_labels"],
        )

    # [D] log-ratio： Reference，Policy 
    chosen_logratio = chosen_logps - ref_chosen_logps
    rejected_logratio = rejected_logps - ref_rejected_logps

    # [E] DPO logits：
    dpo_logits = beta * (chosen_logratio - rejected_logratio)
    chosen_rewards = beta * chosen_logratio.detach()
    rejected_rewards = beta * rejected_logratio.detach()
    reward_margin = chosen_rewards - rejected_rewards

    # [F] DPO loss： chosen  rejected 
    loss = -F.logsigmoid(dpo_logits).mean()
    metrics = {
        "loss": loss.detach(),
        "chosen_reward": chosen_rewards.mean(),
        "rejected_reward": rejected_rewards.mean(),
        "reward_margin": reward_margin.mean(),
        "reward_accuracy": (reward_margin > 0).float().mean(),
    }
    return loss, metrics


# [G] ： Policy， Reference
def train_step(policy_model, ref_model, optimizer, batch, beta=0.1):
    loss, metrics = dpo_forward(policy_model, ref_model, batch, beta=beta)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return metrics


# [H] DPO ： batch 
def train_dpo(policy_model, ref_model, optimizer, dataloader, beta=0.1):
    ref_model.eval()
    for batch in dataloader:
        metrics = train_step(policy_model, ref_model, optimizer, batch, beta)
        print(
            "loss=", float(metrics["loss"]),
            "margin=", float(metrics["reward_margin"]),
        )
