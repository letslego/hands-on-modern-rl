# 10.4 Agent —— SWE-smith 

 Agentic RL 、。：**？**

 Agent  RL "→→"。 bug ——SWE-bench ， 2294 。 Agent。

SWE-smith（Yang et al., 2025）：** bug ， bug**。 Python  task instance， 128  GitHub  50,000+ ——。

## ：、、

SWE-smith ：** bug， bug **。

——""：

1. ****： bug
2. ****：

 2 。 bug ， Agent ""，" bug"。

：

```mermaid
flowchart TD
    A["：GitHub "] --> B["① \nDocker "]
    B --> C["② Bug \n"]
    C --> D["③ \n"]
    D --> E{" ≥1 \n？"}
    E -->|""| F[" task instance"]
    E -->|""| G[""]
    F --> H["④  Issue \nLM "]
    H --> I["：task instance\n（bug patch + F2P  + issue ）"]

    style A fill:#e3f2fd,stroke:#1976d2,color:#000
    style C fill:#fff3e0,stroke:#f57c00,color:#000
    style F fill:#e8f5e9,stroke:#388e3c,color:#000
    style G fill:#ffebee,stroke:#c62828,color:#000
```

。

## ：—— Docker 

 bug ，。，—— Python 、、、……SWE-smith ：

1.  SWE-agent 
2.  SWE-agent ，
3.  Dockerfile， Docker 

。 bug 。

::: tip 
SWE-bench  task instance  Docker ， 1–3 GB。1000  task  1 TB 。SWE-smith ****，1000  task  1 GB—— 1000 。 task instance ，。
:::

## ：Bug ——

， bug。SWE-smith  bug ，。

### ：LM —— bug

 LM ，""。， LM ""，（ bug——）。SWE-smith  LM ""。

```
 LM  prompt（）：

。 Python ：

{function_source}

，。
：
1. 
2. （）
3. 

。
```

 Python  `collections.Counter`  `most_common` ：

```python
# （）
def most_common(self, n=None):
    if n is None:
        return sorted(self.items(), key=_itemgetter(1), reverse=True)
    return _heapq.nlargest(n, self.items(), key=_itemgetter(1))

# LM  bug： reverse=True  False
# ——，
def most_common(self, n=None):
    if n is None:
        return sorted(self.items(), key=_itemgetter(1), reverse=False)  # bug!
    return _heapq.nlargest(n, self.items(), key=_itemgetter(1))
```

 bug ：(1) ，(2) ，(3) ""。LM  bug ， Agent 。

### ：——AST 

 LM，（AST）。SWE-smith  13 ，：

|                        |              |                       | （ bug ） |
| -------------------------- | -------------------- | ----------------------------- | --------------------------- |
| **（Class）**            |  |       | 1.93%                       |
| **（Control Flow）** |          |             | ~30%                        |
|                            |            |                 | ~20%                        |
|                            |          | `if A`  `if not A`        | 47.04%                      |
| **（Expression）**   |            | `+`  `-`，`==`  `!=`      | ~25%                        |
|                            |              | `0`  `1`，`True`  `False` | ~15%                        |
| **（Removal）**        |            |  `pass`  `None` | ~35%                        |
|                            |  return      |  `None`               | ~30%                        |

 `django/utils/dateformat.py` ：

```python
# 
def format(self, format_string):
    result = []
    for token in format_string:
        if token in self.format_mapping:
            result.append(str(self.format_mapping[token]))
        else:
            result.append(token)
    return ''.join(result)

# （invert conditional）
def format(self, format_string):
    result = []
    for token in format_string:
        if token not in self.format_mapping:  # ！
            result.append(str(self.format_mapping[token]))
        else:
            result.append(token)
    return ''.join(result)
```

， token ， token ——。

::: warning 
"" 1.93%  bug——。"" 47.04%——。。
:::

### ：PR —— PR

 PR（Pull Request） review ， bug 。SWE-smith  PR ——" bug "。

```
（PR ）→  bug
    ↓ PR 
 → bug 
    ↓ SWE-smith  PR
 → bug （"" bug）
```

PR  13.18%， SWE-bench  2.46%  5 。 SWE-smith （）， SWE-bench  commit ，。

### ：Bug —— bug  bug

 bug 。SWE-smith  bug ，、 bug：

- ****： bug 
- ****： bug 

 task instance  Agent  bug，。—— bug ，。

## ：——

Bug ， Docker 。 task instance：

- ** 1  "Fail to Pass"（F2P）**：、。 bug 
- ** "Pass to Fail"（P2F）**： bug （）

。、。

## ： Issue 

 bug patch ：。SWE-smith  LM  GitHub Issue ：

- bug  `.diff` patch（）
-  F2P （）
-  bug patch （）

````
 Issue （）：

## Bug: Counter.most_common() returns results in wrong order

### Description
When calling `Counter('abracadabra').most_common(3)`, the method
returns `[('r', 2), ('b', 2), ('a', 5)]` instead of the expected
`[('a', 5), ('b', 2), ('r', 2)]`. The results appear to be sorted
in ascending rather than descending order.

### Reproduction
```python
from collections import Counter
c = Counter('abracadabra')
print(c.most_common(3))
# Expected: [('a', 5), ('b', 2), ('r', 2)]
# Actual:   [('r', 2), ('b', 2), ('a', 5)]
````

### Environment

Python 3.11, collections module

````

， task instance ：(1) bug patch，(2) F2P ，(3) Issue 。 SWE-bench 。

## ：

。 `sympy/sympy`（Python ）。

### 

```bash
#  SWE-smith
pip install swesmith

#  sympy  Docker 
python -m swesmith.build_repo sympy/sympy
````

 `sympy__sympy`  Docker ，。

### LM  Bug

```python
from swesmith.bug_gen.lm import generate_bug_lm

#  sympy  bug
task_instances = generate_bug_lm(
    repo="sympy__sympy",
    target_functions=["simplify", "expand", "factor"],
    model="gpt-4o",
    num_bugs_per_func=5,
)

print(f"Generated {len(task_instances)} candidate bugs")
```

### 

```python
from swesmith.bug_gen.procedural import generate_bug_procedural

#  AST 
task_instances = generate_bug_procedural(
    repo="sympy__sympy",
    strategies=["invert_conditional", "remove_loop", "change_operator"],
    num_candidates=100,
)
```

### 

```python
from swesmith.validate import validate_task_instances

# ， task instance
valid_instances = validate_task_instances(
    repo="sympy__sympy",
    candidates=task_instances,
    docker_image="sympy__sympy",
    timeout=300,  #  5 
)

print(f"Valid: {len(valid_instances)} / {len(task_instances)}")
```

###  Issue

```python
from swesmith.issue_gen import generate_issue

for instance in valid_instances:
    instance.issue_text = generate_issue(
        patch=instance.bug_patch,
        test_source=instance.f2p_test_code,
        test_output=instance.test_output,
    )
```

### ： Task Instance

，：

````python
task_instance = {
    # 
    "repo": "sympy/sympy",
    "instance_id": "sympy__sympy-12345",

    # Bug 
    "bug_patch": """
diff --git a/sympy/simplify/simplify.py b/sympy/simplify/simplify.py
@@ -123,7 +123,7 @@
 def simplify(expr, **kwargs):
-    expr = sympify(expr)
+    expr = expand(expr)  # Bug: should be sympify, not expand
     ...
""",

    # F2P （、）
    "test_patch": """
diff --git a/sympy/simplify/tests/test_simplify.py
+def test_simplify_basic():
+    assert simplify("x^2 + 2*x + 1") == "(x+1)^2"
""",

    # Issue （Agent ）
    "problem_statement": """
## Bug: simplify() incorrectly expands instead of simplifying

When calling `simplify("x^2 + 2*x + 1")`, the result is
`x**2 + 2*x + 1` instead of the expected `(x+1)**2`.

### Reproduction
```python
from sympy import simplify
result = simplify("x^2 + 2*x + 1")
print(result)  # Expected: (x+1)**2, Got: x**2 + 2*x + 1
````

""",
}

````

## 

 bug ：

|  |  | Bug  |  |  |
| --- | --- | --- | --- | --- |
| **LM ** | （~20%） | 、 bug | （ LM ） | 、 bug |
| **** | （2–47%） | 、 | （AST ） |  |
| **PR ** | （~13%） | 、 | （ PR ） |  bug  |
| **Bug ** | （>90%） | 、 | （ bug） |  |

，—— LM  bug， Bug 。

## 

SWE-smith ， 128  Python  50,000+  task instance。 Web （Django、Flask）、（scikit-learn、pandas）、（pytest、pip）。

|  | SWE-bench | SWE-gym | R2E-gym | **SWE-smith** |
| --- | --- | --- | --- | --- |
| Task instance  | 2,294 | 1,283 | 1,468 | **50,000+** |
|  | 12 | 11 | 8 | **128** |
|  | ~2.3 TB | ~1.3 TB | ~1.5 TB | **~128 GB** |
|  |  |  |  | **** |

 SWE-agent-LM-32B  SWE-bench Verified  40.2% Pass@1，。

##  Agent 

。 task instance  Agent  RL ？：

```mermaid
flowchart LR
    A["SWE-smith\nTask Instances"] --> B["Agent Rollout"]
    B --> C["Agent  Issue\n→  bug\n→  patch"]
    C --> D["\n（F2P ）"]
    D --> E{"？"}
    E -->|""| F["reward = 1.0"]
    E -->|""| G["reward = 0.0"]
    F --> H["GRPO / PPO\n"]
    G --> H

    style A fill:#fff3e0,stroke:#f57c00,color:#000
    style H fill:#e3f2fd,stroke:#1976d2,color:#000
````

：

1. **Rollout**：Agent  Issue ， Docker 、 bug、 patch
2. ****： F2P —— RLVR 
3. ****： GRPO  PPO  Agent 

：**SWE-smith  RLVR（）**—— Reward Model， reward。， Reward Hacking 。

## SWE-smith 

### 

** > 。** " bug 、 bug 、"。SWE-smith ：" bug、、"。 Agent ——，（、、）。

**。** Bug —— bug，。SWE-smith ： bug 。，""—— bug， bug 。

**。** "" 1000 ，。

### 

** Python。** SWE-smith  Python 。 JavaScript、Java、C++ 。

**Bug  bug 。**  bug（），。LM  bug ，。

**。** ， bug ， bug 。 SWE-smith ，。

## 

SWE-smith  Agent ：**，**。——" bug →  →  →  Issue"——、、 Python 。

 RLVR（ 9 ）： task instance （）， RL  reward ， Reward Model。

 Agentic RL ，SWE-smith ：**（、、、），**。

::: details 

- Yang J, Lieret K, Jimenez CE, et al. "SWE-smith: Scaling Data for Software Engineering Agents." [arXiv:2504.21798](https://arxiv.org/abs/2504.21798), NeurIPS 2025 D&B Spotlight.
- SWE-smith ：[swesmith.com](https://swesmith.com)
- GitHub ：[SWE-bench/SWE-smith](https://github.com/SWE-bench/SWE-smith)
- ：[SWE-bench/SWE-smith on HuggingFace](https://huggingface.co/datasets/SWE-bench/SWE-smith)（50k+ task instances）
- ：[SWE-agent-LM-32B](https://huggingface.co/SWE-bench/SWE-agent-LM-32B)

  :::
