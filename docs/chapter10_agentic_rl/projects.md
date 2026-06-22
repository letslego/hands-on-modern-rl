---
search: false
---

# ：（）

> 。 [10.4 rLLM DeepCoder ](./rllm-deepcoder-lab)  [10.5 Deep Research Agent](./deep-research-agent)。，。

# ： Agentic RL——、、

 ORM  PRM。。：** 3 （、、），、、。 GRPO ， benchmark 。**

 "→→" 。** agentic RL**——，。

 24GB  GPU（ RTX 4090 / A5000）。 **Qwen2.5-Coder-3B-Instruct** 。

>  MURPHY（multi-turn GRPO for self-correcting code generation）、CodeGym（ICLR 2026， RL ）、VerlTool（ RL ）。 KodCode（）， BigCodeBench-Hard。

```mermaid
flowchart TD
    subgraph Rollout ["① Rollout： + 3 "]
        M1[""] -->|" API"| S[" 1："]
        M1 -->|""| E[" 2："]
        M1 -->|""| T[" 3："]
        S --> O1["：API "]
        E --> O2["：/"]
        T --> O3["：pass/fail "]
        O1 --> M2[""]
        O2 --> M2
        O3 --> M2
        M2 --> M1
    end

    subgraph Collect ["② "]
        TR[":\n[prompt, response₁, obs₁, response₂, obs₂, ...]\n obs "]
    end

    subgraph Train ["③ GRPO RL"]
        G[" →  → \n obs token mask \n token  loss"]
    end

    Rollout --> Collect --> Train

    style M1 fill:#e3f2fd,stroke:#1976d2,color:#000
    style S fill:#fff3e0,stroke:#f57c00,color:#000
    style E fill:#e8f5e9,stroke:#2e7d32,color:#000
    style T fill:#fce4ec,stroke:#c62828,color:#000
    style M2 fill:#e3f2fd,stroke:#1976d2,color:#000
```

##  Agentic RL？

（MURPHY、）：。 "multi-turn RLVR"。

（）：**、、**。 agentic——****。

|                        |  | （） |
| -------------------------- | ------ | ---------------- |
|                      |      |                |
|                |      |                |
|  API       | **** | ****           |
|        | **** | ****           |
| **** |  | **** |

## ：

```bash
pip install torch transformers accelerate datasets
pip install matplotlib numpy peft
```

```python
# ==========================================
# 0. 
# ==========================================
import torch, numpy as np, random, re, os, subprocess, tempfile, warnings
warnings.filterwarnings("ignore")

SEED = 42
MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"
MAX_NEW_TOKENS = 1024
GROUP_SIZE = 4
MAX_EPOCHS = 3
LR = 5e-6
KL_COEFF = 0.05
MAX_TURNS = 5

device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print(f"Device: {device}")
```

## ： +  + 

### 1.1 

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
)
model.eval()
for p in model.parameters():
    p.requires_grad = False
print(f"Model: {MODEL_NAME} ({sum(p.numel() for p in model.parameters())/1e9:.2f}B)")
```

### 1.2  1：API 

，。 Python  + 。

```python
# ==========================================
# 1.2  1：
#      <search>query</search>  API 
# ==========================================

# （ API）
DOC_STORE = {
    "collections": {
        "Counter": "Counter(iterable) -> dict subclass. Count hashable objects. Methods: most_common(n), elements(), update(iterable), subtract(iterable). Example: c = Counter('abcabc'); c.most_common(2) -> [('a', 2), ('b', 2)]",
        "defaultdict": "defaultdict(default_factory) -> dict subclass. Missing keys get default_factory(). Example: d = defaultdict(list); d['k'].append(1)",
        "OrderedDict": "OrderedDict() -> dict subclass that remembers insertion order. Methods: move_to_end(key), popitem(last=True)",
        "deque": "deque(iterable, maxlen=None) -> double-ended queue. Methods: append(x), appendleft(x), pop(), popleft(), rotate(n)",
        "namedtuple": "namedtuple(typename, field_names) -> tuple subclass with named fields. Example: Point = namedtuple('Point', ['x', 'y']); p = Point(1, 2)",
        "ChainMap": "ChainMap(*maps) -> group multiple dicts. Lookups search each dict in order",
    },
    "itertools": {
        "combinations": "combinations(iterable, r) -> iterator of r-length subsequences. Example: list(combinations('ABC', 2)) -> [('A','B'), ('A','C'), ('B','C')]",
        "permutations": "permutations(iterable, r=None) -> iterator of r-length permutations",
        "product": "product(*iterables, repeat=1) -> cartesian product. Example: list(product('AB', '12')) -> [('A','1'),('A','2'),('B','1'),('B','2')]",
        "groupby": "groupby(iterable, key=None) -> consecutive groups. MUST sort first! Example: for k, g in groupby(sorted(data, key=fn), key=fn)",
        "chain": "chain(*iterables) -> chain multiple iterables. chain.from_iterable(iterable) flattens one level",
        "accumulate": "accumulate(iterable, func=operator.add) -> running totals. Example: list(accumulate([1,2,3])) -> [1, 3, 6]",
        "islice": "islice(iterable, start, stop[, step]) -> iterator slicing without creating list",
    },
    "functools": {
        "lru_cache": "@lru_cache(maxsize=128) -> memoization decorator. Example: @lru_cache(maxsize=None)\\ndef fib(n): return n if n < 2 else fib(n-1) + fib(n-2)",
        "reduce": "reduce(function, iterable[, initializer]) -> reduce iterable to single value. Example: reduce(lambda x,y: x+y, [1,2,3]) -> 6",
        "partial": "partial(func, *args, **kwargs) -> fix some arguments. Example: double = partial(operator.mul, 2); double(3) -> 6",
        "cmp_to_key": "cmp_to_key(mycmp) -> convert cmp function to key function for sorted()",
    },
    "re": {
        "findall": "re.findall(pattern, string, flags=0) -> list of all matches. Example: re.findall(r'\\d+', 'a1b22c') -> ['1', '22']",
        "search": "re.search(pattern, string) -> Match object or None. .group() for match, .start()/.end() for position",
        "match": "re.match(pattern, string) -> Match only at beginning of string",
        "sub": "re.sub(pattern, repl, string, count=0) -> substitute. Example: re.sub(r'\\d+', 'N', 'a1b22') -> 'aNbN'",
        "split": "re.split(pattern, string, maxsplit=0) -> split by pattern. Example: re.split(r'[,;]', 'a,b;c') -> ['a', 'b', 'c']",
    },
    "json": {
        "loads": "json.loads(s) -> parse JSON string to Python object. Example: json.loads('{\"a\": 1}') -> {'a': 1}",
        "dumps": "json.dumps(obj, indent=None) -> serialize to JSON string. Example: json.dumps({'a': 1}) -> '{\"a\": 1}'",
        "load": "json.load(fp) -> parse JSON from file object",
        "dump": "json.dump(obj, fp) -> write JSON to file",
    },
    "math": {
        "gcd": "math.gcd(*integers) -> greatest common divisor. Example: math.gcd(12, 8) -> 4",
        "lcm": "math.lcm(*integers) -> least common multiple. Example: math.lcm(4, 6) -> 12",
        "comb": "math.comb(n, k) -> binomial coefficient C(n,k). Example: math.comb(10, 3) -> 120",
        "perm": "math.perm(n, k=None) -> permutations P(n,k)",
        "isqrt": "math.isqrt(n) -> integer square root. Example: math.isqrt(10) -> 3",
        "log": "math.log(x, base=e) -> logarithm. math.log2(x), math.log10(x) also available",
        "ceil": "math.ceil(x) -> smallest integer >= x. math.floor(x) -> largest integer <= x",
    },
    "string": {
        "ascii_lowercase": "string.ascii_lowercase -> 'abcdefghijklmnopqrstuvwxyz'",
        "ascii_uppercase": "string.ascii_uppercase -> 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'",
        "digits": "string.digits -> '0123456789'",
        "ascii_letters": "string.ascii_letters -> ascii_lowercase + ascii_uppercase",
        "Template": "string.Template('$name is $age') -> safe string substitution. .substitute(dict) or .safe_substitute(dict)",
    },
    "heapq": {
        "heappush": "heapq.heappush(heap, item) -> push item onto heap. heap is a plain list",
        "heappop": "heapq.heappop(heap) -> pop smallest item. heapq.heappushpop(heap, item) more efficient",
        "nlargest": "heapq.nlargest(n, iterable, key=None) -> n largest elements. heapq.nsmallest(n, ...) also available",
        "heapify": "heapq.heapify(list) -> transform list into heap in-place in O(n)",
    },
    "bisect": {
        "bisect_left": "bisect.bisect_left(a, x) -> insertion point for x in sorted list a (leftmost). bisect_right for rightmost",
        "insort": "bisect.insort(a, x) -> insert x into sorted list a maintaining order",
    },
    "datetime": {
        "datetime": "datetime.datetime(year, month, day, hour=0, minute=0, second=0). Methods: .strftime(format), .date(), .weekday()",
        "timedelta": "datetime.timedelta(days=0, seconds=0, microseconds=0). Supports +, -, * with datetime",
        "strptime": "datetime.datetime.strptime(date_string, format) -> parse string. Format: %Y, %m, %d, %H, %M, %S",
    },
    "typing": {
        "List": "List[int] -> list of integers. List[str] -> list of strings",
        "Dict": "Dict[str, int] -> dict with string keys and int values",
        "Optional": "Optional[int] -> int or None. Equivalent to Union[int, None]",
        "Tuple": "Tuple[int, str] -> tuple of (int, str). Tuple[int, ...] -> variable length",
    },
    "pathlib": {
        "Path": "Path('dir/file.txt') -> path object. Methods: .read_text(), .write_text(), .exists(), .is_file(), .mkdir(), .glob(pattern)",
    },
    "numpy": {
        "array": "np.array([1,2,3]) -> ndarray. np.zeros(shape), np.ones(shape), np.arange(start,stop,step), np.linspace(start,stop,num)",
        "reshape": "arr.reshape(shape) -> reshape array. -1 infers dimension. arr.flatten() -> 1D copy",
        "argsort": "np.argsort(arr) -> indices that would sort. arr[np.argsort(arr)] == sorted arr",
        "unique": "np.unique(arr, return_counts=False) -> sorted unique values. return_counts=True adds counts",
        "where": "np.where(condition, x, y) -> element-wise conditional. np.where(condition) -> indices where True",
        "dot": "np.dot(a, b) -> matrix multiplication. a @ b is equivalent",
        "sum": "np.sum(arr, axis=None) -> sum. axis=0 sum columns, axis=1 sum rows",
    },
    "pandas": {
        "DataFrame": "pd.DataFrame(data) -> create DataFrame from dict/list. pd.read_csv(path) -> read CSV",
        "groupby": "df.groupby('col') -> GroupBy object. .agg(func), .mean(), .sum(), .count()",
        "merge": "pd.merge(df1, df2, on='key', how='inner') -> join. how: 'left', 'right', 'outer'",
        "value_counts": "series.value_counts() -> frequency of unique values, sorted descending",
        "apply": "df['col'].apply(func) -> apply function to each element. df.apply(func, axis=1) per row",
    },
}


def tool_search(query: str, top_k: int = 3) -> str:
    """
     1：。 <search>query</search> 。
     query  API 。
    """
    query_lower = query.lower()
    results = []
    for module, apis in DOC_STORE.items():
        for api_name, doc in apis.items():
            # 
            score = 0
            for word in query_lower.split():
                if word in api_name.lower():
                    score += 3
                if word in doc.lower():
                    score += 1
                if word in module.lower():
                    score += 2
            if score > 0:
                results.append((score, f"[{module}.{api_name}]\n{doc}"))

    results.sort(key=lambda x: -x[0])
    if not results:
        return "No documentation found. Try different keywords."

    return "\n\n".join(doc for _, doc in results[:top_k])
```

### 1.3  2： +  3：

```python
# ==========================================
# 1.3  2 & 3：
# ==========================================

def tool_execute(code: str, timeout: float = 10.0) -> dict:
    """ 2： Python ， stdout/stderr"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "exec.py")
        with open(path, "w") as f:
            f.write(code)
        try:
            r = subprocess.run(["python", path], capture_output=True, text=True,
                               timeout=timeout, cwd=tmpdir)
            return {
                "success": r.returncode == 0,
                "output": r.stdout.strip()[:500],
                "error": r.stderr.strip()[:300] if r.returncode != 0 else None,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "output": "", "error": "TIMEOUT"}


def tool_test(code: str, test_code: str, timeout: float = 10.0) -> dict:
    """ 3： + ， pass/fail """
    full_code = code + "\n\n" + test_code
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.py")
        with open(path, "w") as f:
            f.write(full_code)
        try:
            r = subprocess.run(["python", path], capture_output=True, text=True,
                               timeout=timeout, cwd=tmpdir)
            if r.returncode == 0:
                return {"passed": True, "detail": "All tests passed", "output": r.stdout.strip()[:200]}
            else:
                # 
                err = r.stderr.strip()
                return {"passed": False, "detail": err[-300:] if err else "unknown error",
                        "output": r.stdout.strip()[:200]}
        except subprocess.TimeoutExpired:
            return {"passed": False, "detail": "TIMEOUT", "output": ""}
```

### 1.4 

```python
from datasets import load_dataset

# ：KodCode（，）
#  MURPHY： 1000  KodCode 
kodcode = load_dataset("KodCode/KodCode-V1", split="train")
#  easy/medium ， 1000 
kodcode_easy = kodcode.filter(lambda x: x["gpt_difficulty"] in ["easy", "medium"])
train_data = list(kodcode_easy.shuffle(seed=SEED).select(range(1000)))
print(f"Training data: {len(train_data)} KodCode problems (easy/medium)")

# ：BigCodeBench-Hard（148 ，）
# BigCodeBench  HumanEval ， API 
eval_data = load_dataset("bigcode/bigcodebench-hard", split="v0.1.4")
print(f"Evaluation data: {len(eval_data)} BigCodeBench-Hard problems")
```

## ：Agent Prompt 

````python
# ==========================================
# 2. Agent System Prompt： 3 
# ==========================================

AGENT_PROMPT = """You are a Python expert. You have 3 tools available:

1. **Search documentation**: Use when you need to look up an unfamiliar API.
   Format: <search>your query</search>
   Example: <search>Counter most_common</search>

2. **Execute code**: Run Python code to test ideas or compute results.
   Format:
   ```python
   # your code
   print(result)
   ```

3. **Run tests**: Submit your final solution to be tested.
   Format:
   ```submit
   def your_solution(...):
       ...
   ```

Strategy: Search docs BEFORE writing code if you're unsure about an API.
Test your code with tool 2 before submitting with tool 3.
You can use tools in any order and multiple times."""

SEARCH_PATTERN = re.compile(r'<search>(.*?)</search>', re.DOTALL)
CODE_PATTERN = re.compile(r'```(?:python|py)?\n(.*?)\n```', re.DOTALL)
SUBMIT_PATTERN = re.compile(r'```submit\n(.*?)\n```', re.DOTALL)
PAD_ID = tokenizer.pad_token_id
````

```python
def parse_tool_calls(text: str) -> list:
    """"""
    calls = []
    # ：
    for m in SEARCH_PATTERN.finditer(text):
        calls.append(("search", m.group(1).strip(), m.start()))
    for m in CODE_PATTERN.finditer(text):
        calls.append(("execute", m.group(1).strip(), m.start()))
    for m in SUBMIT_PATTERN.finditer(text):
        calls.append(("submit", m.group(1).strip(), m.start()))
    calls.sort(key=lambda x: x[2])
    return calls
```

## ： Agent Rollout

```python
# ==========================================
# 3.  Agent Rollout（）
#     3 
# ==========================================

def run_multi_tool_rollout(task_prompt: str, task_test: str, entry_point: str,
                           temperature=0.7, max_turns=MAX_TURNS, verbose=False):
    """
     Agent Rollout。

    （ Search-R1/VerlTool/CodeGym）：
    1. （//）
    2. token 
    3.  observation  token  mask（ loss）
    4.  token  loss
    """
    init_messages = [
        {"role": "system", "content": AGENT_PROMPT},
        {"role": "user", "content": task_prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(init_messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    real_ids = list(prompt_ids)
    masked_ids = list(prompt_ids)  # prompt 

    submitted_code = None
    test_passed = False
    is_valid = True
    tool_usage = {"search": 0, "execute": 0, "submit": 0, "none": 0}

    for turn_idx in range(max_turns):
        input_t = torch.tensor([real_ids], dtype=torch.long).to(device)
        attn_t = torch.ones_like(input_t)

        with torch.no_grad():
            out = model.generate(
                input_ids=input_t, attention_mask=attn_t,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=temperature, do_sample=temperature > 0,
                top_p=0.95, pad_token_id=PAD_ID
            )

        gen_ids = out[0][input_t.shape[1]:].tolist()
        response_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        #  token（real  masked ）
        real_ids.extend(gen_ids)
        masked_ids.extend(gen_ids)

        # ---  ---
        calls = parse_tool_calls(response_text)

        if not calls:
            tool_usage["none"] += 1
            continue

        # （）
        tool_type, tool_input, _ = calls[-1]
        tool_usage[tool_type] += 1

        # --- ， observation ---
        if tool_type == "search":
            doc_result = tool_search(tool_input)
            obs_text = f"\n<doc_result>\n{doc_result[:500]}\n</doc_result>\n"
            if verbose:
                print(f"  Turn {turn_idx+1} [SEARCH]: {tool_input[:40]}")

        elif tool_type == "execute":
            exec_result = tool_execute(tool_input)
            if exec_result["success"]:
                obs_text = f"\n<exec_result>\n{exec_result['output'][:300]}\n</exec_result>\n"
            else:
                err = exec_result.get("error", "unknown") or "unknown"
                obs_text = f"\n<exec_result>\nERROR: {err[:200]}\n</exec_result>\n"
            if verbose:
                status = "OK" if exec_result["success"] else "ERR"
                print(f"  Turn {turn_idx+1} [EXEC]: {status}")

        elif tool_type == "submit":
            submitted_code = tool_input
            # 
            test_result = tool_test(tool_input, task_test, entry_point)
            test_passed = test_result["passed"]
            detail = test_result["detail"][:300]
            obs_text = f"\n<test_result>\n{'PASS' if test_passed else 'FAIL'}: {detail}\n</test_result>\n"
            if not test_passed:
                obs_text += "Fix your code and submit again.\n"
            if verbose:
                print(f"  Turn {turn_idx+1} [SUBMIT]: {'PASS' if test_passed else 'FAIL'}")

        obs_ids = tokenizer.encode(obs_text, add_special_tokens=False)
        # observation token：real ，masked  PAD
        real_ids.extend(obs_ids)
        masked_ids.extend([PAD_ID] * len(obs_ids))

        # ，
        if tool_type == "submit" and test_passed:
            break

    # 
    if submitted_code:
        final_test = tool_test(submitted_code, task_test, entry_point)
        test_passed = final_test["passed"]
    else:
        test_passed = False

    #  info_mask
    ids_tensor = torch.tensor([real_ids], dtype=torch.long)
    masked_tensor = torch.tensor([masked_ids], dtype=torch.long)
    info_mask = (masked_tensor != PAD_ID).long()
    labels = ids_tensor.clone()
    labels[info_mask == 0] = -100

    return {
        "input_ids": ids_tensor,
        "attention_mask": torch.ones_like(ids_tensor),
        "labels": labels,
        "info_mask": info_mask,
        "submitted_code": submitted_code,
        "passed": test_passed,
        "is_valid": is_valid,
        "turns": turn_idx + 1,
        "tool_usage": tool_usage,
    }
```

### 3.1  Rollout

```python
#  rollout
print("Sanity check — Multi-Tool Agent Rollout:")
print("-" * 60)
for i in [0, 5, 10]:
    item = eval_data[i]
    result = run_multi_tool_rollout(
        item["instruct_prompt"], item["test"], item["entry_point"],
        temperature=0.3, verbose=True
    )
    n_assist = result["info_mask"].sum().item()
    n_tool = (result["info_mask"] == 0).sum().item()
    print(f"  {item['task_id']}: {'PASS' if result['passed'] else 'FAIL'} "
          f"(turns: {result['turns']}, LLM: {n_assist}, tool: {n_tool}, "
          f"tools used: {result['tool_usage']})")
print("-" * 60)
```

## ：—— vs  vs 

 baseline，。

```python
# ==========================================
# 4. ：
# ==========================================
EVAL_N = 50  #  50 

def eval_no_tools(model_to_eval, data, n):
    """，"""
    model_to_eval.eval()
    results = []
    for i, item in enumerate(list(data)[:n]):
        messages = [{"role": "user", "content": item["instruct_prompt"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model_to_eval.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                         temperature=0.0, pad_token_id=PAD_ID)
        code = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        test_result = tool_test(code, item["test"], item["entry_point"])
        results.append(test_result["passed"])
        if (i+1) % 25 == 0:
            print(f"  [No-Tools] {i+1}/{n}: {sum(results)/len(results):.1%}")
    return sum(results) / len(results)

def eval_code_only(model_to_eval, data, n):
    """（， MURPHY）"""
    model_to_eval.eval()
    results = []
    for i, item in enumerate(list(data)[:n]):
        result = run_multi_tool_rollout(
            item["instruct_prompt"], item["test"], item["entry_point"],
            temperature=0.0, max_turns=3
        )
        results.append(result["passed"])
        if (i+1) % 25 == 0:
            print(f"  [Code-Only] {i+1}/{n}: {sum(results)/len(results):.1%}")
    return sum(results) / len(results)

def eval_multi_tool(model_to_eval, data, n, label="Multi-Tool"):
    """3 """
    model_to_eval.eval()
    results = []
    for i, item in enumerate(list(data)[:n]):
        result = run_multi_tool_rollout(
            item["instruct_prompt"], item["test"], item["entry_point"],
            temperature=0.0, max_turns=5
        )
        results.append(result["passed"])
        if (i+1) % 25 == 0:
            print(f"  [{label}] {i+1}/{n}: {sum(results)/len(results):.1%}")
    return sum(results) / len(results)

print("BASELINE EVALUATION (before training)")
print("=" * 60)
baseline_notools = eval_no_tools(model, eval_data, EVAL_N)
baseline_codeonly = eval_code_only(model, eval_data, EVAL_N)
baseline_multitool = eval_multi_tool(model, eval_data, EVAL_N, "Baseline-MultiTool")
print(f"\n  No tools:     {baseline_notools:.1%}")
print(f"  Code only:    {baseline_codeonly:.1%}")
print(f"  Multi-tool:   {baseline_multitool:.1%}")
print("=" * 60)
```

## ： Rollout + GRPO RL 

```python
# ==========================================
# 5. GRPO RL ：
# ==========================================
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW

#  + reference model
model_rl = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
)
model_rl.enable_input_require_grads()
model_rl = get_peft_model(model_rl, LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
    lora_dropout=0.05, target_modules=["q_proj", "v_proj"],
))

ref_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

optimizer_rl = AdamW(filter(lambda p: p.requires_grad, model_rl.parameters()), lr=LR)

training_log = {"epoch": [], "pass_rate": [], "mean_reward": [], "loss": [],
                "search_rate": [], "submit_rate": []}

#  demo
TRAIN_SUBSET = list(range(0, 500, 5))  # 100 
train_subset = [train_data[i] for i in TRAIN_SUBSET]

print("=" * 60)
print("GRPO RL Training — Multi-Tool Agent")
print("=" * 60)

for epoch in range(MAX_EPOCHS):
    model_rl.train()
    epoch_rewards, epoch_losses = [], []
    epoch_passed = 0
    epoch_tool_stats = {"search": 0, "execute": 0, "submit": 0, "none": 0}
    random.shuffle(train_subset)

    for task_idx, item in enumerate(train_subset):
        # KodCode  test 
        task_test = item.get("test", "")
        task_prompt = item["question"]
        #  entry_point（KodCode ）
        entry_point = "solution"

        # ---- Phase 1: On-Policy Rollout ----
        trajectories = []
        for g in range(GROUP_SIZE):
            result = run_multi_tool_rollout(
                task_prompt, task_test, entry_point,
                temperature=0.7, max_turns=MAX_TURNS
            )
            # Reward ：
            # 1. Outcome reward：
            reward = 1.0 if result["passed"] else 0.0
            # 2.  shaping：，（）
            if not result["passed"] and result["tool_usage"]["search"] > 0:
                reward = 0.05 * min(result["tool_usage"]["search"], 3)
            # 3. ，
            if not result["submitted_code"]:
                reward = -0.1
            result["reward"] = reward
            trajectories.append(result)

        # ---- Phase 2: GRPO Advantage ----
        rewards = np.array([t["reward"] for t in trajectories])
        mean_r, std_r = rewards.mean(), rewards.std() + 1e-8
        advantages = (rewards - mean_r) / std_r

        # ---- Phase 3:  ----
        for traj, advantage in zip(trajectories, advantages):
            input_ids = traj["input_ids"].to(device)
            attention_mask = traj["attention_mask"].to(device)
            labels = traj["labels"].to(device)
            info_mask = traj["info_mask"].to(device)

            if (info_mask == 1).sum() == 0:
                continue

            outputs = model_rl(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            policy_lp = -outputs.loss

            with torch.no_grad():
                ref_out = ref_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                ref_lp = -ref_out.loss

            kl = policy_lp - ref_lp
            loss = -advantage * policy_lp + KL_COEFF * kl

            optimizer_rl.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_rl.parameters(), 1.0)
            optimizer_rl.step()
            epoch_losses.append(loss.item())

        epoch_rewards.extend(rewards.tolist())
        epoch_passed += sum(1 for t in trajectories if t["passed"])
        for t in trajectories:
            for k in epoch_tool_stats:
                epoch_tool_stats[k] += t["tool_usage"].get(k, 0)

        if (task_idx + 1) % 25 == 0:
            pr = epoch_passed / ((task_idx+1) * GROUP_SIZE)
            total_tools = sum(epoch_tool_stats.values())
            sr = epoch_tool_stats["search"] / max(total_tools, 1)
            print(f"  Epoch {epoch+1} | Task {task_idx+1}/{len(train_subset)} | "
                  f"Pass: {pr:.1%} | Search rate: {sr:.1%}")

    # Epoch summary
    pr = epoch_passed / (len(train_subset) * GROUP_SIZE)
    total_tools = sum(epoch_tool_stats.values())
    training_log["epoch"].append(epoch+1)
    training_log["pass_rate"].append(pr)
    training_log["mean_reward"].append(np.mean(epoch_rewards))
    training_log["loss"].append(np.mean(epoch_losses) if epoch_losses else 0)
    training_log["search_rate"].append(epoch_tool_stats["search"] / max(total_tools, 1))
    training_log["submit_rate"].append(epoch_tool_stats["submit"] / max(total_tools, 1))
    print(f"  Epoch {epoch+1} Summary: Pass={pr:.1%}, Reward={np.mean(epoch_rewards):.3f}, "
          f"Loss={training_log['loss'][-1]:.4f}, SearchRate={training_log['search_rate'][-1]:.1%}")

model_rl.eval()
```

## ：——？

```python
# ==========================================
# 6. 
# ==========================================

print("=" * 60)
print("POST-TRAINING Evaluation on BigCodeBench-Hard")
print("=" * 60)

rl_notools = eval_no_tools(model_rl, eval_data, EVAL_N)
rl_multitool = eval_multi_tool(model_rl, eval_data, EVAL_N, "RL-MultiTool")

print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)
print(f"  {'Method':<25} {'Pass@1':>8}")
print(f"  {'-'*33}")
print(f"  {'Baseline (no tools)':<25} {baseline_notools:>7.1%}")
print(f"  {'Baseline (code only)':<25} {baseline_codeonly:>7.1%}")
print(f"  {'Baseline (multi-tool)':<25} {baseline_multitool:>7.1%}")
print(f"  {'RL (no tools)':<25} {rl_notools:>7.1%}")
print(f"  {'RL (multi-tool)':<25} {rl_multitool:>7.1%}")
print(f"  {'-'*33}")
print(f"  {'Multi-tool gain':<25} {rl_multitool - baseline_notools:>+7.1%}")
print(f"  {'RL tool-learning gain':<25} {rl_multitool - baseline_multitool:>+7.1%}")
print("=" * 60)
```

### 

```python
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- ：Pass@1  ---
ax = axes[0]
methods = ['No Tools\n(Baseline)', 'Code Only\n(Baseline)', 'Multi-Tool\n(Baseline)', 'Multi-Tool\n(After RL)']
pass_rates = [baseline_notools, baseline_codeonly, baseline_multitool, rl_multitool]
colors = ['#bdbdbd', '#90a4ae', '#42a5f5', '#66bb6a']

bars = ax.bar(methods, pass_rates, color=colors, edgecolor='#333', linewidth=1.5)
ax.set_ylabel('pass@1')
ax.set_title(f'BigCodeBench-Hard (n={EVAL_N})', fontweight='bold')
ax.set_ylim(0, max(max(pass_rates) * 1.5, 0.2))

for bar, v in zip(bars, pass_rates):
    ax.text(bar.get_x() + bar.get_width()/2., v + 0.01,
            f'{v:.1%}', ha='center', fontsize=12, fontweight='bold')

best = np.argmax(pass_rates)
if pass_rates[best] > pass_rates[0]:
    ax.annotate(f'+{(pass_rates[best]-pass_rates[0])*100:.1f}pp',
                xy=(best, pass_rates[best]), xytext=(best, pass_rates[best]+0.06),
                fontsize=13, fontweight='bold', color='#2e7d32',
                arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=2))

# --- ： ---
ax = axes[1]
epochs = training_log["epoch"]
ax.plot(epochs, [p*100 for p in training_log["pass_rate"]], 'o-', color='#388e3c', lw=2, label='Pass Rate (%)')
ax.plot(epochs, training_log["mean_reward"], 's--', color='#1976d2', lw=2, label='Mean Reward')
ax.set_xlabel('Epoch')
ax.set_title('GRPO RL Training', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# --- ： ---
ax = axes[2]
ax.plot(epochs, [s*100 for s in training_log["search_rate"]], 'o-', color='#f57c00', lw=2, label='Search Rate (%)')
ax.plot(epochs, [s*100 for s in training_log["submit_rate"]], 's-', color='#c62828', lw=2, label='Submit Rate (%)')
ax.set_xlabel('Epoch')
ax.set_title('Tool Selection Strategy', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('Multi-Tool Agentic RL: Qwen2.5-Coder-3B on BigCodeBench-Hard', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("multi_tool_agentic_rl.png", dpi=150)
print("Saved: multi_tool_agentic_rl.png")
```

## ： Agentic RL 

|                     |                    |  Benchmark                     |                            |
| ----------------------- | ---------------------- | ---------------------------------- | ------------------------------ |
| **MURPHY**              |  +         | HumanEval, MBPP, BigCodeBench-Hard | Multi-turn GRPO，1.7B  +8% |
| **CodeGym** (ICLR'26)   |          | OOD                        |  RL                |
| **VerlTool**            | //SQL//SWE | AIME, NQ, Spider, SWE-Verified     |                    |
| **Search-R1**           |                | NQ, HotpotQA, TriviaQA             |  token mask                |
| **SimpleTIR** (ICLR'26) |                | AIME 2024                          | Void turn                  |
| **RLEF** (Meta)         |  +         | CodeContests                       | 8B  SOTA           |

## 

****：

1.  **3 **（、、）
2. ****、、
3.  GRPO RL ****
4.  BigCodeBench-Hard 

****：

- **3 **（vs ）：，
- ****：BigCodeBench-Hard  Python ，
- **Token mask**： observation token  loss（`labels = -100`）
- **Reward shaping**： = 1.0， = 0.05\*n（）， = -0.1
- ****：KodCode （ BigCodeBench ），1000 （MURPHY ）
- ****： KodCode， BigCodeBench-Hard

****：

- RL ，（ ""）
-  pass@1  no-tool baseline
-  "Tool Selection Strategy" 

 Agentic RL ——[](./tool-use-agents)。

---

# ： Deep Research Agent

 RL 、、 Web Agent / Code Agent 。****——Deep Research Agent（）。 AI ，、、，。

2025-2026 ，Deep Research Agent  Agentic RL 。、、、、、。

##  Deep Research Agent？

Deep Research Agent " + "。：** AI 、，、？** 、、、。

 Web Agent ，Deep Research Agent ：

|      | Web Agent                      | Deep Research Agent                        |
| -------- | ------------------------------ | ------------------------------------------ |
|  | （、） | （、、） |
|  |  3-10                    |  20-100+                             |
|  | /                  |  +  +          |
|  | 、             | 、、               |

###  vs  API：

Deep Research Agent ，：

****—— AI ，、、。 DeepResearcher（ RL ）[^deepresearcher]、WebAgent-R1（）。、，、。

** API **—— API  JSON 。 OpenResearcher（，）[^openresearcher]、PokeeResearch-7B（ API ）。、、，。

。—— Tongyi DeepResearch  Search（ API）、Visit（）、Python Interpreter  [^tongyi_dr]。

## ： ReAct 

Deep Research Agent 。，：

1. **ReAct：**
   -  Thought → Action → Observation。
   - ：、、。
   - ""。

2. **Iterative Research：**
   - """"， ReAct 。
   - " →  →  →  → "。
   - ，、。

3. **Multi-agent Synthesis：**
   - ，，、、、。
   - ，""""，。
   - DeepResearcher、Fathom-DeepResearch 。

：**ReAct ，iterative research ，multi-agent synthesis 。** Agentic RL ，，、、。

## 

 Deep Research 。 LLM """"。

### DeepResearcher： RL 

DeepResearcher **、** RL  [^deepresearcher]。 RAG ， prompt ——DeepResearcher ，。

："（Browsing Agents）"，。（RLVR），。

**：。**  DeepResearcher —— RL ，，****：

1. **（Planning）**：，
2. **（Cross-verification）**：，
3. ****：，
4. ****：，

 RL  agent ""——****。 Agentic RL ： SFT ， RL 。

### Tongyi DeepResearch：Agentic Mid-training + Post-training

 Tongyi DeepResearch  Deep Research  [^tongyi_dr]。 benchmark  OpenAI o3、DeepSeek-V3.1（671B）， 30.5B—— MoE（Mixture of Experts）， 3.3B ，。

**。** Tongyi DeepResearch  **Agentic Mid-training + Post-training** ：

1. **Agentic Mid-training（Agentic CPT）**：。： 32K  agentic ， 128K （64K-128K）agentic 。""，**agentic **——，""。，。

2. **Agentic Post-training**：——SFT （）、on-policy RL（ GRPO +）、（）。

**。** ，Tongyi DeepResearch ：

- **Context Management **：。Tongyi  Context Management——，""。。
- ****：。Mid-training ""（、）""（、）；Post-training  RL ，。 API 、、。

 BrowseComp、WebWalkerQA、FRAMES、HLE  benchmark  SOTA [^tongyi_dr]。

### PokeeResearch-7B：

PokeeResearch-7B  Deep Research ， 7B  [^pokeeresearch]。：****。

：""，（、、），7B ，。 Deep Research Agent —— A100 ， GPU 。

### SFR-DeepResearch：

Salesforce  SFR-DeepResearch ：****（Autonomous Single Agent）[^sfr_dr]。、、，。

****——。：、、，。SFR ****（、 RL ） RL  agent ，。

### rStar2-Agent：

rStar2-Agent  RL  [^rstar2]。 GRPO  agent RL  14B ，：**，**。

：， 100B+ ，rStar2-Agent —— RL （、）， 14B 。

```mermaid
flowchart LR
    subgraph "Deep Research Agent "
        A["Agentic Mid-training\n(Tongyi)"] --> B["SFT\n()"]
        B --> C["RL "]
        C --> D["\n(、)"]
    end

    subgraph ""
        E["\n(DeepResearcher)"]
        F[" API\n(OpenResearcher)"]
        G["\n(Tongyi)"]
    end

    C --- E
    C --- F
    C --- G

    style A fill:#e3f2fd,stroke:#1976d2,color:#000
    style D fill:#e8f5e9,stroke:#388e3c,color:#000
```

## ：""

 Deep Research ，""——， reward 。、。

### ：CaRR

****：Deep Research Agent 、""，""——， URL，。 outcome reward（）。

****： AI  Citation-aware Rubric Rewards（CaRR）[^carr_dr]  RL 。，，：

1. **Rubric **：（Rubrics）， Rubric 。
2. ****： Rubric 。
3. ****： URL（ 20 ），， Rubric 。
4. ****：， Rubric 。

 Rubric  Rubric 。（） $\alpha$ ， GRPO 。

****：CaRR ""——，、，"→→"。

### ：Atom-Searcher

****：Deep Research 。 reward（=1，=0），（credit assignment）——，。

****：Atom-Searcher **（Atomic Thought Reward, ATR）**[^atom_searcher]，，。： reward，""。

**""""？**  ATR ""。（" A  B"），。， token 。

****：ATR 。，。， ATR ， reward ——：，。

### ：DR Tulu

****：RL ——**Reward Hacking**。""，。""，""。，""。

****：Allen AI  DR Tulu  **RLER（Reinforcement Learning with Evolving Rubrics）**[^dr_tulu]——。""：

1. ****： Rubrics 。""，
2. ****：，。""
3. ****：。""

，""，。

****：RLER ""——，。 CaRR 、Web-Shepherd 。

###  RL：Memento

****：RL 、、。，。 Agent ？

****：Memento  [^memento]——****，""（Episodic Memory） Agent 。：

1. ****：
2. ****：，
3. ****：，

**？** Memento  GAIA （87.88% Pass@3）， RL 。：**""""**。，RL  Agent ——。，Memento  RL 。

### ：Web-Shepherd

****：，outcome reward（）。 Agent  30 ， 28 ，——outcome reward ，。

****：Web-Shepherd **（PRM）**  [^web_shepherd]。 ORM（Outcome Reward Model），PRM ，。

****：Web-Shepherd  PRM ， outcome reward 、。

****：PRM  10.9 。，""，——****。

****：Web-Shepherd  PRM  Atom-Searcher  ATR （），——PRM ，ATR 。。

## ：RL ""

、 Deep Research Agent ，。。

### OpenResearcher：

****： Deep Research Agent ，、API 、。。

****：OpenResearcher **、** [^openresearcher]。，""：`search`（）、`open`（）、`find`（）。，、。

****：OpenResearcher  97K ， 100+ 。。

****：，OpenResearcher —— API key， GPU ，。：、。

### Tongyi DeepResearch ：、

Tongyi DeepResearch  [^tongyi_dr] ，。**、**，：

- **Mid-training **： agent ，。：
  - ****：，（、）
  - ****：——
  - ****：，，
  - ****：，

- **Post-training **：，（），， PhD 

**""**：。，，。，。

### Fathom-DeepResearch：

****：""—— GPT-4 ，。

****：Fathom-DeepResearch ****（Multi-agent Self-play） DUETQA  [^fathom_dr]。 4B ：

- **（Fathom-Search-4B）**：
- **（Fathom-Synthesizer-4B）**：

——，，、。

****：Fathom  GAN（）——。，。"" agent 。

## ："" Deep Research？

>  Deep Research 。 Agentic （、、 benchmark ） [10.3 ：、 Badcase](./industrial-evaluation)。

Deep Research Agent ""。 Deep Research ：

|        |                  |                          |
| ---------- | -------------------- | -------------------------------- |
|  |      | （Exact Match/F1） |
|  |  |  URL  +    |
|  |  |  PRM                   |
|    |  |            |

：

- **GAIA**：，、。
- **Humanity's Last Exam (HLE)**：，。
- **BrowseComp / BrowseComp-ZH**： seeking ，、、。
- **WebWalkerQA**：，""。
- **FRAMES**：，""。
- **xbench-DeepSearch**：，。
- **WebArena / Mind2Web**：，。
- **BFCL**：/API ，。

 benchmark ，：

- ****：GAIA、HLE、FRAMES、xbench-DeepSearch
- ****：BrowseComp、BrowseComp-ZH、WebWalkerQA
- ****：WebArena、Mind2Web、BFCL

 Deep Research Agent ：""，""，""。，，，。

### ？

""， RL ：

- ****：、URL 
- ****：，
- ****：，
- ****：， token 
- ****：，

## ：

，：

**——：**

```python
#  reward：
reward = 1.0 if answer == ground_truth else 0.0
```

**——：**

```python
# 
reward = (
    accuracy_score(answer, ground_truth)      # 
    + 0.2 * valid_tool_call_ratio             # 
    - 0.1 * (num_turns / max_turns)           # 
)
```

**——：**

```python
#  +  + 
reward = (
    0.4 * accuracy_score(answer, ground_truth)
    + 0.3 * citation_quality_score(answer)    #  + 
    + 0.2 * cross_validation_score(answer)    # 
    + 0.1 * efficiency_bonus(num_turns)       # 
)
```

## 

|          |      |                                             |
| ------------ | -------- | --------------------------------------------------- |
| Awesome-GRPO |    |  GRPO  RL                         |
| LLM-Explorer |  | ， RL ， 37.27% |
| WebSailor-V2 |  |  RL  Agent  |
| ReLook       |  |  LLM  RL，      |

## 

 Deep Research Agent，：

1. **DeepResearcher**： RL ，""。
2. **OpenResearcher**：， Deep Research 。
3. **rStar2-Agent**： RL ，。

## ：Deep Research 

""""——Deep Research """"。 Deep Research ****：。、、， Agent 。

###  RL 

、""， RL ：

**。** 、、、。 trade-off——。

**。**  3000-10000 ， RLHF （500-1000 ）。。

**。** ——、、。。

###  RL：LongWriter-Zero

LongWriter-Zero[^longwriter] ：，****。：

```python
def longwriter_reward(text, prompt):
    """ reward"""
    # 1. （）
    target = extract_target_length(prompt)
    length_reward = compute_length_reward(len(text), target)

    # 2. （ RM ）
    quality_reward = writing_quality_model.score(text)

    # 3. （、、）
    structure_reward = evaluate_structure(text)

    return 0.3 * length_reward + 0.4 * quality_reward + 0.3 * structure_reward
```

：**RL **。 SFT ， reward 。

Writer-R1[^writerr1] ****—— Memory-augmented Replay Policy Optimization，""""，，。

### 

RL-Struct[^rlstruct] ****，：

|     |                                |      |
| ------- | -------------------------------------- | ------------ |
| Level 0 | （ JSON/Markdown）   |  = 0   |
| Level 1 |                          |  |
| Level 2 | （，） |  |
| Level 3 | （、）                 | RM   |
| Level 4 | （、）                 | RM   |

（ 0 ），（RM ）。，。

###  Reward 

：

```python
def report_reward(report, task, verified_facts=None):
    """ reward"""
    accuracy = accuracy_reward(report, verified_facts or {})
    structure = structure_reward(report)
    citation = citation_reward(report)
    length = length_reward(len(report), task.target_length)
    relevance = compute_relevance(report, task.question)

    return (
        0.30 * accuracy +
        0.20 * structure +
        0.15 * citation +
        0.10 * length +
        0.25 * relevance
    )
```

****—— 500 ， 5000 。 10.2  HardGen[^hardgen] 。

### Deep Research  RL

 Deep Research ：

```
 1:  RL
  → 、、
  → reward:  + 

 2:  RL
  → 、、
  → reward:  +  + 
```

——""，""。， RL 。

## ： Rubrics  Search Agent RL 

、、。，：**， RL  AI  Agent？**  Reward Model ， RL 。

### Step 1： AI  Rubrics

Rubrics（）""。 AI  Agent ：

|        |                      |                 |
| ---------- | ------------------------ | ----------------------- |
|  |          |  + LLM    |
|  |        |       |
|    |          | URL  +  |
|  |  |           |
|      |            |             |

 1-5 ，""：1  = ，3  = ，5  = 。

### Step 2： Rubrics  Reward Model

 Rubrics， Reward Model。

**。**  query，（）。（ LLM-as-Judge） Rubrics ，——" A  B "。

**RM 。**  Bradley-Terry （ 8 ） Reward Model。 (query, search_result) ，。 RM  RL  reward 。

：** RM， Rubrics  RM？**

 RM ， credit assignment。 RM ，。， RM ， RM。

```python
def train_search_reward_model(preference_data, base_model):
    """ Reward Model"""
    # preference_data: [(query, result_better, result_worse), ...]
    #  Bradley-Terry 
    # loss = -log(sigmoid(rm(query, better) - rm(query, worse)))

    rm = RewardModel(base_model)
    for query, better, worse in preference_data:
        score_better = rm.score(query, better)
        score_worse = rm.score(query, worse)
        loss = -torch.log(torch.sigmoid(score_better - score_worse))
        loss.backward()
        rm.update()
    return rm
```

### Step 3： RL  Search Agent

 RM， RL 。 GRPO （ Critic）：

```python
async def search_agent_grpo_step(model, rm, queries, group_size=4, max_turns=10):
    """Search Agent  GRPO """
    all_groups = []

    for query in queries:
        trajectories = []
        for _ in range(group_size):
            # Rollout: Agent 
            result = await rollout_search_agent(model, query, max_turns)
            #  RM 
            reward = rm.score(query, result.final_answer)
            #  Rubrics  reward
            reward += 0.2 * citation_bonus(result)       # 
            reward += 0.1 * efficiency_bonus(result)      # 
            reward -= 0.3 * hallucination_penalty(result)  # 
            trajectories.append((result, reward))

        # 
        trajectories.sort(key=lambda x: x[1], reverse=True)
        all_groups.append(trajectories)

    # GRPO 
    for group in all_groups:
        best, worst = group[0], group[-1]
        if best[1] > worst[1]:
            await model.grpo_update(
                prompt=best[0].prompt,
                chosen=best[0].trajectory,
                rejected=worst[0].trajectory,
                advantage=best[1] - worst[1]
            )

    return all_groups
```

### Step 4：Reward Hacking 

RL  **Reward Hacking**——" reward "，。：

- ****：" reward "， 3-4 （）
- ****： ground truth ，
- ****：""，

**。** （）。 RM ，， Reward Hacking 。

**。** DR Tulu[^rler_dr]  RLER（）—— Rubrics ""，，""。，CaRR[^carr_dr] ——，。

### Step 5：

（），：

**。** ：、、。，""。

**。** ——""、""。

**。** ""（、）""。

"Rubrics → RM → RL → Hacking  → "。 Rubrics、 RM、 RL  reward 。

## ： Deep Research Agent

，"" Deep Research Agent  RL 。。

### ： →  →  → 

 Deep Research Agent ：

```mermaid
flowchart LR
    Q[""] --> S["\n（Search API）"]
    S --> R["\n（）"]
    R --> T["\n（LLM ）"]
    T --> D{"？"}
    D -->|""| S
    D -->|""| A[""]

    style S fill:#fff3e0,stroke:#e65100
    style T fill:#e3f2fd,stroke:#1976d2
    style A fill:#e8f5e9,stroke:#2e7d32
```

### ： Agent 

```python
# ==========================================
#  Deep Research Agent 
# ==========================================

import json
import requests

class ResearchEnvironment:
    """Deep Research Agent """

    def __init__(self, search_api_key=None):
        self.search_api_key = search_api_key
        self.max_turns = 10  #  10 

    def step(self, state, action):
        """ Agent ，"""
        if action["type"] == "search":
            #  API（ Serper API / Tavily API ）
            results = self._search(action["query"])
            return state + f"\n: {results}"

        elif action["type"] == "read":
            # （ Jina Reader API ）
            content = self._read_url(action["url"])
            return state + f"\n: {content[:2000]}"

        elif action["type"] == "answer":
            # Agent 
            return state, action["content"], True  # done=True

        return state, "", False

    def _search(self, query):
        """ API"""
        #  API 
        #  Tavily: https://tavily.com
        #  Serper: https://serper.dev
        return f"[ '{query}' ]"

    def _read_url(self, url):
        """"""
        return f"[{url} ]"

    def evaluate(self, question, answer, ground_truth):
        """"""
        #  reward：
        if answer.strip() == ground_truth.strip():
            return 1.0
        # ：
        key_facts = ground_truth.split("，")
        covered = sum(1 for f in key_facts if f in answer)
        return covered / len(key_facts) if key_facts else 0.0
```

### ：

Agent ——、：

```python
def format_agent_prompt(question, history, turn):
    """ Agent  prompt"""
    return f"""。。

：
- search(query): 
- read(url): 
- answer(content): 

 {turn}/{10} 。

: {question}

:
{history}

（JSON ）:
{{"type": "search", "query": "..."}}

{{"type": "read", "url": "..."}}

{{"type": "answer", "content": "..."}}
"""
```

### ：GRPO 

 GRPO  +  Agent ：

```python
import asyncio

async def rollout_one(model, env, question, max_turns=10):
    """ Agent  rollout"""
    state = ""
    for turn in range(max_turns):
        prompt = format_agent_prompt(question, state, turn)
        action_text = await model.generate_async(prompt)

        #  Agent 
        try:
            action = json.loads(action_text)
        except:
            action = {"type": "answer", "content": action_text}

        # 
        state, answer, done = env.step(state, action)
        if done:
            break

    #  reward
    reward = env.evaluate(question, answer, ground_truth)
    return {"state": state, "answer": answer, "reward": reward}

async def grpo_train_step(model, env, questions, ground_truths, group_size=4):
    """GRPO ： + """
    all_trajectories = []

    #  group_size 
    for q, gt in zip(questions, ground_truths):
        trajectories = []
        for _ in range(group_size):
            traj = await rollout_one(model, env, q)
            trajectories.append(traj)

        # ： reward 
        trajectories.sort(key=lambda t: t["reward"], reverse=True)
        all_trajectories.append(trajectories)

    # GRPO ： reward ， reward 
    for group in all_trajectories:
        best = group[0]   # reward 
        worst = group[-1] # reward 

        if best["reward"] > worst["reward"]:
            # 
            #  GRPO Loss  DPO Loss
            await model.update(
                prompt=best["state"],
                chosen=best["answer"],
                rejected=worst["answer"]
            )

    return all_trajectories
```

### ：

```python
# ：
train_data = [
    {"question": "2024 ？？",
     "answer": "John Hopfield  Geoffrey Hinton，"},
    {"question": "GRPO  PPO  LLM ？",
     "answer": "GRPO  Critic ，"},
    # ... 
]

# 
for epoch in range(3):
    batch = train_data[epoch::3]  # 
    trajectories = await grpo_train_step(
        model, env,
        [d["question"] for d in batch],
        [d["answer"] for d in batch],
        group_size=4
    )
    avg_reward = sum(t[0]["reward"] for t in trajectories) / len(trajectories)
    print(f"Epoch {epoch}:  reward = {avg_reward:.2f}")
```

### 

。，：

1. ** API**： Tavily  Serper API，
2. ** reward**：、（[](#)）
3. ****：[Agentic ](./tool-use-agents) rollout
4. ****：[10.2 ](./tool-use-and-trajectory)
5. ****： DeepResearcher  rStar2-Agent 

<details>
<summary>：Deep Research Agent ， RLVR、PPO、GRPO ？</summary>

Deep Research Agent  RL ：

- **RLVR（ 9 ）**：Deep Research  reward ""—— URL 、、。， Reward Model。
- **GRPO（ 9 ）**：DeepResearcher  + ， GRPO 。
- **PPO（ 7 ）**： PPO  RL ， Value Function  credit assignment 。
- **PRM vs ORM（10.1 ）**：CaRR、Atom-Searcher、Web-Shepherd  Deep Research  ORM（） PRM（）。：，PRM 。

Deep Research Agent  RL ""—— reward  credit assignment，，。

</details>

## 

### 、 Deep Research 

"→→"，： LLM ， RL 。****（mid-training vs  post-training）、****（ vs  vs ）、****（ vs  vs MoE）。

[^deepresearcher]: Zheng Y, et al. "DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments." [arXiv:2504.03160](https://arxiv.org/abs/2504.03160), EMNLP 2025. ****： RL 。RL 、、，——"RL "。

[^tongyi_dr]: Tongyi DeepResearch Team. "Tongyi DeepResearch Technical Report." [arXiv:2510.24701](https://arxiv.org/abs/2510.24701), 2025. ****： Agentic Mid-training + Post-training ， Mid-training  agentic ， agent 。30.5B MoE（3.3B ） benchmark  SOTA， MoE  agent 。

[^sfr_dr]: Nguyen X-P, et al. "SFR-DeepResearch: Towards Effective Reinforcement Learning for Autonomously Reasoning Single Agents." [arXiv:2509.06283](https://arxiv.org/abs/2509.06283), 2025. ****：Salesforce ，——，。 RL  agent 。

[^pokeeresearch]: PokeeResearch-7B. [HuggingFace Model Card](https://huggingface.co/PokeeAI/pokee_research_7b), 2025. ****：7B ， Deep Research 。。

### 、

， Deep Research RL ：****。：""（outcome reward），。****（ vs ）****（ vs  vs ）。

[^carr_dr]: Zhang J, Lv X, Feng L, Hou L, Li J. "Chaining the Evidence: Robust Reinforcement Learning for Deep Search Agents with Citation-Aware Rubric Rewards." [arXiv:2601.06021](https://arxiv.org/abs/2601.06021), 2026. ****： AI 。 Rubric，，"" Deep Research 。

[^atom_searcher]: Deng Y, et al. "Atom-Searcher: Enhancing Agentic Deep Research via Fine-Grained Atomic Thought Reward." [arXiv:2508.12800](https://arxiv.org/abs/2508.12800), 2025. ****：（ATR），。 RL ——， reward ，ATR 。

[^dr_tulu]: Shao R, Asai A, et al. "DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research." [arXiv:2511.19399](https://arxiv.org/abs/2511.19399), 2025. ****：Allen AI 。RLER ——，。"" Reward Hacking：，。

[^rler_dr]: Shao R, Asai A, et al. "DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research." [arXiv:2511.19399](https://arxiv.org/abs/2511.19399), 2025. ， RL ， Reward Hacking。

[^web_shepherd]: Chae H, et al. "Web-Shepherd: Advancing PRMs for Reinforcing Web Agents." [arXiv:2505.15277](https://arxiv.org/abs/2505.15277), NeurIPS 2025 Spotlight. ****：（PRM）， WebAgent  10.9 ， agent 。

[^rstar2]: Shang N, et al. "rStar2-Agent: Agentic Reasoning Technical Report." [arXiv:2508.20722](https://arxiv.org/abs/2508.20722), 2025. ****： GRPO  Agent RL ， 14B 。—— RL 。

### 、

 Deep Research RL ""——、、。：，。****，（ vs  vs ）。

[^openresearcher]: Li Z, Jiang D, et al. "OpenResearcher: A Fully Open Pipeline for Long-Horizon Deep Research Trajectory Synthesis." [arXiv:2603.20278](https://arxiv.org/abs/2603.20278), 2026. ****：——97K+ ，，（search/open/find）。。

[^fathom_dr]: Singh S, Singh K, Moturi P. "Fathom-DeepResearch: Unlocking Long Horizon Information Retrieval and Synthesis for SLMs." [arXiv:2509.24107](https://arxiv.org/abs/2509.24107), 2025. ****： 4B """"， DUETQA 。：，。

[^hardgen]: Hao B, et al. "From Failure to Mastery: Generating Hard Samples for Tool-use Agents." [arXiv:2601.01498](https://arxiv.org/abs/2601.01498), 2026. ****：。""——，，。

### 、 RL

 Deep Research ""——。：（3000-10000 ）、、。**** RL ，。

[^longwriter]: Wu Y, et al. "LongWriter-Zero: Mastering Ultra-Long Text Generation via Reinforcement Learning." [arXiv:2506.18841](https://arxiv.org/abs/2506.18841), 2025. ****： RL ——， reward（++）。

[^writerr1]: Zhao J, et al. "Writer-R1: Enhancing Generative Writing in LLMs via Memory-augmented Replay Policy Optimization." [arXiv:2603.15061](https://arxiv.org/abs/2603.15061), 2026. ****： Memory-augmented Replay Policy Optimization，""""，。

[^rlstruct]: Hu R, Wu S. "RL-Struct: A Lightweight Reinforcement Learning Framework for Reliable Structured Output in LLMs." [arXiv:2512.00319](https://arxiv.org/abs/2512.00319), 2025. ****：，，（ 0 ），（RM ）。，。

### 

[^memento]: Zhou H, et al. "Memento: Fine-tuning LLM Agents without Fine-tuning LLMs." [arXiv:2508.16153](https://arxiv.org/abs/2508.16153), 2025. ****：Memento ——****， Agent 。 GAIA （87.88% Pass@3），：""""。，RL  Agent ，。

， 9 。，——[](../chapter12_future_trends/intro)， RL 。
