# 9.5 ： GRPO  API

 RLVR：， Reward Model。：， API，，。

。“”，“”。：

> What was AAPL's closing price on 2025-01-03?

，：

```json
{
  "name": "get_stock_price",
  "arguments": { "ticker": "AAPL", "date": "2025-01-03" }
}
```

：， API 。，：、、JSON 。 API ，，、、。

 AWS  financial tool-calling GRPO + TRL [^aws-financial-tool-grpo]，： API ， verifier ， TRL  `GRPOTrainer` 。，“ API  RLVR”。

## ： API 

。 API：

|                |                        |                                  |
| ------------------ | -------------------------- | ---------------------------------------- |
| `get_stock_price`  |  | `ticker`, `date`                         |
| `get_revenue`      |  | `company`, `fiscal_year`                 |
| `convert_currency` |          | `amount`, `from_currency`, `to_currency` |

，。：

|                                       |               |
| --------------------------------------------- | --------------------- |
| “Get MSFT close price on 2025-01-02.”         |  `get_stock_price`  |
| “What was Tesla revenue in fiscal year 2024?” |  `get_revenue`      |
| “Convert 120 USD to EUR.”                     |  `convert_currency` |
| “Write a poem about markets.”                 |             |

 RLVR  verifier： JSON ，、、 schema、。“”，“”。

```mermaid
flowchart LR
    U[""] --> M[" πθ"]
    M --> C[" tool call JSON"]
    C --> V["Verifier\nJSON + schema + execution"]
    V --> R["reward"]
    R --> G["GRPO "]
    G --> M

    style V fill:#e8f5e9,stroke:#2e7d32
    style R fill:#fff3e0,stroke:#f57c00
```

## ：

，： API 、、 query-call 。，。：

- `prompt`：。
- `gold_call`：。
- `expected_result`：。

```python
import json
from datasets import Dataset

TOOLS = [
    {
        "name": "get_stock_price",
        "description": "Get the closing stock price for a ticker on a date.",
        "parameters": {
            "ticker": {"type": "string"},
            "date": {"type": "string", "format": "YYYY-MM-DD"},
        },
        "required": ["ticker", "date"],
    },
    {
        "name": "get_revenue",
        "description": "Get annual revenue for a company and fiscal year.",
        "parameters": {
            "company": {"type": "string"},
            "fiscal_year": {"type": "integer"},
        },
        "required": ["company", "fiscal_year"],
    },
    {
        "name": "convert_currency",
        "description": "Convert money between currencies.",
        "parameters": {
            "amount": {"type": "number"},
            "from_currency": {"type": "string"},
            "to_currency": {"type": "string"},
        },
        "required": ["amount", "from_currency", "to_currency"],
    },
]

STOCK_DB = {
    ("AAPL", "2025-01-02"): 243.85,
    ("AAPL", "2025-01-03"): 243.36,
    ("MSFT", "2025-01-02"): 418.58,
    ("MSFT", "2025-01-03"): 423.35,
}

REVENUE_DB = {
    ("Apple", 2024): 391_035_000_000,
    ("Microsoft", 2024): 245_122_000_000,
    ("Tesla", 2024): 97_690_000_000,
}

FX_DB = {
    ("USD", "EUR"): 0.92,
    ("EUR", "USD"): 1.09,
    ("USD", "JPY"): 157.2,
}


def execute_tool(name: str, arguments: dict):
    if name == "get_stock_price":
        return STOCK_DB[(arguments["ticker"], arguments["date"])]
    if name == "get_revenue":
        return REVENUE_DB[(arguments["company"], arguments["fiscal_year"])]
    if name == "convert_currency":
        rate = FX_DB[(arguments["from_currency"], arguments["to_currency"])]
        return round(arguments["amount"] * rate, 2)
    raise ValueError(f"Unknown tool: {name}")


def build_prompt(user_query: str) -> str:
    return (
        "You are a financial assistant. Choose exactly one tool call if a tool "
        "is needed. Return only JSON with this shape: "
        '{"name": "...", "arguments": {...}}. '
        "If no tool is needed, return "
        '{"name": "no_call", "arguments": {}}.\n\n'
        f"Available tools:\n{json.dumps(TOOLS, ensure_ascii=False, indent=2)}\n\n"
        f"User request: {user_query}\n"
    )


def make_dataset() -> Dataset:
    rows = []
    examples = [
        (
            "What was AAPL's closing price on 2025-01-03?",
            {"name": "get_stock_price",
             "arguments": {"ticker": "AAPL", "date": "2025-01-03"}},
        ),
        (
            "Get MSFT close price on 2025-01-02.",
            {"name": "get_stock_price",
             "arguments": {"ticker": "MSFT", "date": "2025-01-02"}},
        ),
        (
            "How much revenue did Tesla report in fiscal year 2024?",
            {"name": "get_revenue",
             "arguments": {"company": "Tesla", "fiscal_year": 2024}},
        ),
        (
            "Convert 120 USD to EUR.",
            {"name": "convert_currency",
             "arguments": {"amount": 120, "from_currency": "USD",
                           "to_currency": "EUR"}},
        ),
    ]

    for query, gold_call in examples:
        rows.append({
            "prompt": build_prompt(query),
            "gold_call": json.dumps(gold_call, ensure_ascii=False),
            "expected_result": str(execute_tool(
                gold_call["name"], gold_call["arguments"]
            )),
        })

    rows.append({
        "prompt": build_prompt("Write a short poem about financial markets."),
        "gold_call": json.dumps({"name": "no_call", "arguments": {}}),
        "expected_result": "no_call",
    })
    return Dataset.from_list(rows)
```

，。 query ，：、、、。AWS ： API ， GRPO 。

## Reward：

 RLVR “”。，，。

|            |                |   |
| ---------------- | ------------------------ | ----- |
| JSON       |        | `0.2` |
|        |  API             | `0.3` |
|  schema  | 、 | `0.3` |
|      |    | `0.2` |

，。 RLVR verifier。

```python
def parse_call(text: str) -> dict | None:
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def schema_ok(call: dict, gold: dict) -> bool:
    if call.get("name") != gold.get("name"):
        return False
    if not isinstance(call.get("arguments"), dict):
        return False
    for key, gold_value in gold["arguments"].items():
        if key not in call["arguments"]:
            return False
        if not isinstance(call["arguments"][key], type(gold_value)):
            return False
    return True


def tool_reward(completions, gold_call, expected_result, **kwargs):
    rewards = []
    for completion, gold_raw, expected in zip(
        completions, gold_call, expected_result
    ):
        gold = json.loads(gold_raw)
        call = parse_call(completion)
        if call is None:
            rewards.append(0.0)
            continue

        reward = 0.2
        if call.get("name") == gold["name"]:
            reward += 0.3
        if schema_ok(call, gold):
            reward += 0.3

        if gold["name"] == "no_call":
            if call.get("name") == "no_call":
                reward += 0.2
            rewards.append(reward)
            continue

        try:
            result = execute_tool(call["name"], call["arguments"])
            if str(result) == expected:
                reward += 0.2
        except Exception:
            pass

        rewards.append(reward)
    return rewards
```

， reward  API 。“”，、、。，****。

## ： TRL  GRPOTrainer

 verifier， GRPO。。，， `Qwen/Qwen2.5-0.5B-Instruct`； AWS ， `Qwen/Qwen3-1.7B`。

```python
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer


dataset = make_dataset()

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

training_args = GRPOConfig(
    output_dir="outputs/financial-tool-grpo",
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=4,
    max_prompt_length=2048,
    max_completion_length=256,
    temperature=0.8,
    logging_steps=1,
    save_steps=50,
    max_steps=100,
)

trainer = GRPOTrainer(
    model="Qwen/Qwen3-1.7B",
    args=training_args,
    train_dataset=dataset,
    reward_funcs=tool_reward,
    peft_config=peft_config,
)

trainer.train()
```

 RLVR ：

1.  prompt  `num_generations=4` 。
2. `tool_reward()` 。
3. GRPO 。
4. ，。

 reward ：， JSON、schema 。

## ： reward

： reward ， API。 reward ， JSON，。

：

```python
def evaluate_tool_call(predictions: list[str], examples: list[dict]) -> dict:
    total = len(examples)
    valid_json = 0
    name_match = 0
    schema_match = 0
    exact_match = 0

    for pred, ex in zip(predictions, examples):
        gold = json.loads(ex["gold_call"])
        call = parse_call(pred)
        if call is None:
            continue

        valid_json += 1
        if call.get("name") == gold["name"]:
            name_match += 1
        if schema_ok(call, gold):
            schema_match += 1

        try:
            if gold["name"] == "no_call":
                if call.get("name") == "no_call":
                    exact_match += 1
            else:
                result = execute_tool(call["name"], call["arguments"])
                if str(result) == ex["expected_result"]:
                    exact_match += 1
        except Exception:
            pass

    return {
        "response_validity": valid_json / total,
        "function_name_accuracy": name_match / total,
        "schema_match": schema_match / total,
        "exact_match": exact_match / total,
    }
```

：

- `response_validity`：。
- `function_name_accuracy`：。
- `schema_match`：。
- `exact_match`：，。

AWS  financial tool-calling ：Qwen3-1.7B  GRPO/RLVR ，exact match  `0.62`  `0.96`，response validity  `0.78`  `0.99`，schema match  `0.90`  `0.95`[^aws-financial-tool-grpo]。： API ，“”、。

##  SFT 

： SFT？ JSON，？

SFT ，。 SFT “”，GRPO/RLVR “”。：

|   |                            |                                  |
| --------- | ---------------------------------- | ---------------------------------------------- |
| SFT       |  JSON 、   | 、、 |
| GRPO/RLVR | 、 | verifier  reward hacking         |

： SFT ， GRPO/RLVR 。 A  B， B  A 。

## 

**， JSON 。**  JSON ，，。

**， no-call 。**  API。，。

**，。** ，。，`get_revenue`、`get_profit`、`get_cash_flow` ，。

**， reward 。**  API ， API 、mock  dry-run。RL ，。

## 

 RLVR 。， verifier ：

-  RLVR ；
-  RLVR ；
-  RLVR  JSON、schema、。

 tool-calling  RLVR 。 API  schema、、； verifier。， GRPO 。

[^aws-financial-tool-grpo]: AWS Builder Center, [Fine-tune Small Language Models for Production-Grade Tool Calling with GRPO using Hugging Face TRL on Amazon SageMaker AI](https://builder.aws.com/content/35x6VR6kZYSn3JgNQmcNmIVK32Y/fine-tune-small-language-models-for-production-grade-tool-calling-with-grpo-using-hugging-face-trl-on-amazon-sagemaker-ai).
