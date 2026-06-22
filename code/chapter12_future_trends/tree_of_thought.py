"""
13： (Tree of Thought) 
—— LLM 

Tree of Thought (ToT) ：
- （）
- 
-  top-k 
- 

"24"，：
1. Chain-of-Thought (CoT): 
2. Tree of Thought (ToT):  + 
3. Random: 

：
    python tree_of_thought.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from copy import deepcopy

# 
os.makedirs("output", exist_ok=True)

# 
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# ：24
# ==========================================
class TwentyFourGame:
    """
    24

    ：
        - 4（1~13）
        -  +, -, *, / 
        - 
        - 24

    ""：
        - 
        - ，""
        - 
    """

    # 
    OPERATIONS = ['+', '-', '*', '/']

    @staticmethod
    def apply_op(a, b, op):
        """，； None"""
        if op == '+':
            return a + b
        elif op == '-':
            return a - b
        elif op == '*':
            return a * b
        elif op == '/':
            if abs(b) < 1e-10:
                return None  # 
            return a / b

    @staticmethod
    def is_close_to_target(value, target=24.0, tol=1e-6):
        """"""
        return abs(value - target) < tol


# ==========================================
# ：
# ==========================================
class ThoughtNode:
    """
    

    ""：
        - numbers: 
        - history: 
        - score: （）
        - parent: 
        - children: 
    """

    def __init__(self, numbers, history=None, parent=None):
        self.numbers = list(numbers)  # 
        self.history = history or []  # 
        self.parent = parent
        self.children = []
        self.score = 0.0  # 

    def is_terminal(self):
        """（）"""
        return len(self.numbers) <= 1

    def get_expression(self):
        """"""
        return ' → '.join(self.history) if self.history else ''

    def __repr__(self):
        return f"Node(nums={self.numbers}, score={self.score:.2f})"


def evaluate_node(node, target=24.0):
    """
    

    ：
        - （1）：24
        - （）：""

     LLM  ToT ""。
     LLM ，。
    """
    if node.is_terminal():
        # ：24
        diff = abs(node.numbers[0] - target)
        return max(0.0, 1.0 - diff / 50.0)  # 

    # ：24""
    # ：
    nums = node.numbers
    best_potential = 0.0

    for i in range(len(nums)):
        for j in range(len(nums)):
            if i == j:
                continue
            for op in TwentyFourGame.OPERATIONS:
                result = TwentyFourGame.apply_op(nums[i], nums[j], op)
                if result is None:
                    continue
                # 
                new_nums = [nums[k] for k in range(len(nums)) if k != i and k != j]
                new_nums.append(result)

                if len(new_nums) == 1:
                    # ，24
                    potential = max(0.0, 1.0 - abs(new_nums[0] - target) / 50.0)
                else:
                    # ：
                    potential = 0.5 * max(0.0, 1.0 - abs(result - target) / 50.0)
                best_potential = max(best_potential, potential)

    return best_potential


def generate_children(node):
    """
    

    ，
    。
    """
    children = []
    nums = node.numbers
    n = len(nums)

    if n < 2:
        return children

    for i in range(n):
        for j in range(n):
            if i == j:
                continue  # 
            for op in TwentyFourGame.OPERATIONS:
                result = TwentyFourGame.apply_op(nums[i], nums[j], op)
                if result is None:
                    continue  # 

                # 
                new_nums = [nums[k] for k in range(n) if k != i and k != j]
                new_nums.append(result)

                # 
                step = f"{nums[i]} {op} {nums[j]} = {result:.2f}"
                new_history = node.history + [step]

                child = ThoughtNode(new_nums, new_history, parent=node)
                children.append(child)

    return children


# ==========================================
# ：
# ==========================================
def search_tree_of_thought(numbers, breadth=3, max_depth=4, target=24.0, verbose=True):
    """
    Tree of Thought 

    ：
        1. （）
        2. 
        3.  breadth 
        4. 

    ：
        numbers: 
        breadth: （）
        max_depth: 
        target: 
        verbose: 

    ：
        best_node: 
        tree_data: （）
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Tree of Thought  (breadth={breadth})")
        print(f"  : {numbers}, : {target}")
        print(f"{'='*60}")

    root = ThoughtNode(numbers)
    current_beam = [root]  # 
    tree_data = {'nodes_per_level': [], 'scores_per_level': []}
    total_evaluated = 0

    for depth in range(max_depth):
        if verbose:
            print(f"\n---  {depth + 1}  ---")

        # ：，
        all_children = []
        for node in current_beam:
            children = generate_children(node)
            # 
            for child in children:
                child.score = evaluate_node(child, target)
                total_evaluated += 1
            all_children.extend(children)

        if not all_children:
            if verbose:
                print("  ")
            break

        # ：， top-k
        all_children.sort(key=lambda x: x.score, reverse=True)
        current_beam = all_children[:breadth]

        # 
        level_nodes = [c.get_expression() for c in current_beam]
        level_scores = [c.score for c in current_beam]
        tree_data['nodes_per_level'].append(level_nodes)
        tree_data['scores_per_level'].append(level_scores)

        if verbose:
            print(f"   {len(all_children)} （ {total_evaluated} ）")
            print(f"   top-{breadth}:")
            for idx, node in enumerate(current_beam):
                nums_str = ', '.join([f"{n:.1f}" for n in node.numbers])
                print(f"    [{idx+1}] ={node.score:.3f} | =[{nums_str}] | {node.get_expression()}")

        # 
        for node in current_beam:
            if (node.is_terminal()
                    and TwentyFourGame.is_close_to_target(node.numbers[0], target)):
                if verbose:
                    print(f"\n  *** ！***")
                    print(f"   = {node.numbers[0]:.4f}")
                    print(f"  : {node.get_expression()}")
                return node, tree_data

    # ，
    best_node = max(current_beam, key=lambda x: x.score)
    if verbose:
        if best_node.is_terminal():
            print(f"\n  ，:")
            print(f"   = {best_node.numbers[0]:.4f}")
            print(f"  : {best_node.get_expression()}")
        else:
            print(f"\n  （）")
            print(f"  : {best_node}")

    return best_node, tree_data


def search_chain_of_thought(numbers, target=24.0, verbose=True):
    """
    Chain-of-Thought (CoT) 

    ：
        （），
         breadth=1  ToT。

     CoT ""。
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Chain-of-Thought  ()")
        print(f"  : {numbers}, : {target}")
        print(f"{'='*60}")

    # CoT  breadth=1  ToT
    result, _ = search_tree_of_thought(
        numbers, breadth=1, max_depth=4, target=target, verbose=verbose
    )
    return result


def search_random(numbers, n_trials=50, target=24.0, verbose=True):
    """
    （）

    ：
        ，
        ，。
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"   ( {n_trials} )")
        print(f"  : {numbers}, : {target}")
        print(f"{'='*60}")

    best_result = None
    best_diff = float('inf')
    best_history = []
    successes = 0

    for trial in range(n_trials):
        remaining = list(numbers)
        history = []

        for step in range(3):  # 43
            if len(remaining) < 2:
                break

            # 
            indices = np.random.choice(len(remaining), 2, replace=False)
            i, j = indices
            op = np.random.choice(TwentyFourGame.OPERATIONS)

            result = TwentyFourGame.apply_op(remaining[i], remaining[j], op)
            if result is None:
                break

            step_str = f"{remaining[i]} {op} {remaining[j]} = {result:.2f}"
            history.append(step_str)

            # 
            new_remaining = [remaining[k] for k in range(len(remaining))
                             if k != i and k != j]
            new_remaining.append(result)
            remaining = new_remaining

        if len(remaining) == 1:
            diff = abs(remaining[0] - target)
            if diff < best_diff:
                best_diff = diff
                best_result = remaining[0]
                best_history = history
            if TwentyFourGame.is_close_to_target(remaining[0], target):
                successes += 1

    if verbose:
        if best_result is not None:
            print(f"  : {best_result:.4f} (={best_diff:.4f})")
            print(f"  : {successes}/{n_trials} ")
        else:
            print(f"  ")

    return best_result, best_diff, successes


# ==========================================
# ：
# ==========================================
def visualize_search_tree(tree_data, title="Tree of Thought "):
    """
    

    ，
    。
    """
    n_levels = len(tree_data['nodes_per_level'])
    if n_levels == 0:
        print("")
        return

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # 
    y_spacing = 1.0
    max_score = 0.0

    for level_scores in tree_data['scores_per_level']:
        for s in level_scores:
            max_score = max(max_score, s)

    for level in range(n_levels):
        nodes = tree_data['nodes_per_level'][level]
        scores = tree_data['scores_per_level'][level]
        n_nodes = len(nodes)

        #  y 
        y = (n_levels - 1 - level) * y_spacing

        # 
        x_positions = np.linspace(0.5, n_nodes - 0.5, n_nodes)

        for i, (node_expr, score) in enumerate(zip(nodes, scores)):
            x = x_positions[i] if n_nodes > 1 else 0.5

            # 
            size = 200 + score * 800
            color_val = score / max(max_score, 0.01)
            color = plt.cm.RdYlGn(color_val)

            # 
            ax.scatter(x, y, s=size, c=[color], edgecolors='black',
                       linewidths=1.5, zorder=5)

            # 
            ax.text(x, y + 0.15, f'{score:.2f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

            # （）
            expr_short = node_expr.split(' → ')[-1] if ' → ' in node_expr else node_expr
            ax.text(x, y - 0.15, expr_short, ha='center', va='top',
                    fontsize=7, color='#333333',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow',
                              alpha=0.7))

        # 
        if level > 0:
            prev_nodes = tree_data['nodes_per_level'][level - 1]
            prev_scores = tree_data['scores_per_level'][level - 1]
            prev_n = len(prev_nodes)
            prev_x = np.linspace(0.5, prev_n - 0.5, prev_n) if prev_n > 1 else [0.5]
            prev_y = (n_levels - level) * y_spacing

            for pi in range(prev_n):
                for ci in range(n_nodes):
                    ax.plot([prev_x[pi], x_positions[ci]],
                            [prev_y, y],
                            color='gray', alpha=0.2, linewidth=0.8, zorder=1)

    # 
    ax.set_ylabel('', fontsize=12)
    ax.set_xticks([])
    y_ticks = [(n_levels - 1 - l) * y_spacing for l in range(n_levels)]
    y_labels = [f'{l+1}' for l in range(n_levels)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def visualize_comparison(results, problem_labels):
    """
    

    ：
        results: ，
        problem_labels: 
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(' vs  vs  — ', fontsize=16, fontweight='bold')

    colors = ['#4CAF50', '#2196F3', '#FF9800']
    strategies = ['Tree of Thought', 'Chain of Thought', 'Random']
    keys = ['tot', 'cot', 'random']

    # ---- 1： ----
    ax1 = axes[0]
    success_rates = [results[k]['success_rate'] for k in keys]
    bars = ax1.bar(strategies, success_rates, color=colors, edgecolor='black', linewidth=0.8)
    ax1.set_ylabel('', fontsize=12)
    ax1.set_title('', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    for bar, rate in zip(bars, success_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{rate:.0%}', ha='center', fontsize=11, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)

    # ---- 2： ----
    ax2 = axes[1]
    avg_errors = [results[k]['avg_error'] for k in keys]
    bars = ax2.bar(strategies, avg_errors, color=colors, edgecolor='black', linewidth=0.8)
    ax2.set_ylabel('', fontsize=12)
    ax2.set_title('(24)', fontsize=13, fontweight='bold')
    for bar, err in zip(bars, avg_errors):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f'{err:.2f}', ha='center', fontsize=11, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)

    # ---- 3： ----
    ax3 = axes[2]
    breadths = [1, 2, 3, 5, 8]
    # 
    success_by_breadth = results['tot']['breadth_success_rates']
    ax3.plot(breadths[:len(success_by_breadth)], success_by_breadth,
             'o-', color='#4CAF50', linewidth=2.5, markersize=10, label='ToT ')
    ax3.axhline(y=results['cot']['success_rate'], color='#2196F3',
                linestyle='--', linewidth=2, label='CoT (breadth=1)')
    ax3.axhline(y=results['random']['success_rate'], color='#FF9800',
                linestyle='--', linewidth=2, label='Random')
    ax3.set_xlabel(' (breadth)', fontsize=12)
    ax3.set_ylabel('', fontsize=12)
    ax3.set_title(' ToT ', fontsize=13, fontweight='bold')
    ax3.set_ylim(0, 1.1)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(breadths[:len(success_by_breadth)])

    plt.tight_layout()
    return fig


# ==========================================
# ：
# ==========================================
def run_experiment():
    """
    

    ：
        1. 24
        2.  ToT、CoT、Random 
        3. 、
        4.  ToT 
        5. 
    """

    # 24（）
    problems = [
        [1, 2, 3, 4],
        [2, 3, 4, 6],
        [1, 5, 5, 5],
        [3, 3, 8, 8],
        [4, 4, 10, 10],
        [1, 4, 5, 6],
        [2, 6, 7, 7],
        [3, 6, 6, 8],
        [2, 3, 5, 7],
        [1, 3, 4, 6],
    ]

    target = 24.0

    print("=" * 60)
    print("  13： (Tree of Thought) ")
    print("=" * 60)
    print(f"  : 24 — 4 {target}")
    print(f"  : {len(problems)}")
    print(f"  : Tree of Thought, Chain of Thought, Random")
    print("-" * 60)

    # ---- 1: Tree of Thought (breadth=3) ----
    print("\n" + "=" * 60)
    print("  1: Tree of Thought (breadth=3)")
    print("=" * 60)

    tot_successes = 0
    tot_errors = []
    # 
    first_tree_data = None

    for idx, nums in enumerate(problems):
        result, tree_data = search_tree_of_thought(
            nums, breadth=3, max_depth=4, target=target, verbose=(idx == 0)
        )
        if idx == 0:
            first_tree_data = tree_data

        if result.is_terminal():
            error = abs(result.numbers[0] - target)
            tot_errors.append(error)
            if TwentyFourGame.is_close_to_target(result.numbers[0], target):
                tot_successes += 1
                print(f"   {idx+1} {nums}: ! ={result.numbers[0]:.2f}")
            else:
                print(f"   {idx+1} {nums}: , ={result.numbers[0]:.2f} (={error:.2f})")
        else:
            tot_errors.append(abs(target))  # 
            print(f"   {idx+1} {nums}: ")

    tot_success_rate = tot_successes / len(problems)
    tot_avg_error = np.mean(tot_errors)

    # ---- 2: Chain of Thought (breadth=1) ----
    print("\n" + "=" * 60)
    print("  2: Chain of Thought ()")
    print("=" * 60)

    cot_successes = 0
    cot_errors = []

    for idx, nums in enumerate(problems):
        result = search_chain_of_thought(nums, target=target, verbose=(idx == 0))
        if result.is_terminal():
            error = abs(result.numbers[0] - target)
            cot_errors.append(error)
            if TwentyFourGame.is_close_to_target(result.numbers[0], target):
                cot_successes += 1
                print(f"   {idx+1} {nums}: ! ={result.numbers[0]:.2f}")
            else:
                print(f"   {idx+1} {nums}: , ={result.numbers[0]:.2f} (={error:.2f})")
        else:
            cot_errors.append(abs(target))
            print(f"   {idx+1} {nums}: ")

    cot_success_rate = cot_successes / len(problems)
    cot_avg_error = np.mean(cot_errors)

    # ---- 3: Random ----
    print("\n" + "=" * 60)
    print("  3: ")
    print("=" * 60)

    random_successes = 0
    random_errors = []

    for idx, nums in enumerate(problems):
        best_result, best_diff, successes = search_random(
            nums, n_trials=50, target=target, verbose=(idx == 0)
        )
        if best_result is not None:
            random_errors.append(best_diff)
            if TwentyFourGame.is_close_to_target(best_result, target):
                random_successes += 1
                print(f"   {idx+1} {nums}: ! ={best_result:.2f}")
            else:
                print(f"   {idx+1} {nums}: ={best_result:.2f} (={best_diff:.2f})")
        else:
            random_errors.append(abs(target))
            print(f"   {idx+1} {nums}: ")

    random_success_rate = random_successes / len(problems)
    random_avg_error = np.mean(random_errors)

    # ----  ToT  ----
    print("\n" + "=" * 60)
    print("   ToT ")
    print("=" * 60)

    breadth_values = [1, 2, 3, 5, 8]
    breadth_success_rates = []

    for b in breadth_values:
        successes = 0
        for nums in problems:
            result, _ = search_tree_of_thought(
                nums, breadth=b, max_depth=4, target=target, verbose=False
            )
            if (result.is_terminal()
                    and TwentyFourGame.is_close_to_target(result.numbers[0], target)):
                successes += 1
        rate = successes / len(problems)
        breadth_success_rates.append(rate)
        print(f"  breadth={b}:  = {rate:.0%} ({successes}/{len(problems)})")

    # ----  ----
    print("\n" + "=" * 60)
    print("  ")
    print("=" * 60)
    print(f"  {'':<25} {'':>10} {'':>12}")
    print(f"  {'-'*47}")
    print(f"  {'Tree of Thought (b=3)':<25} {tot_success_rate:>9.0%} {tot_avg_error:>12.2f}")
    print(f"  {'Chain of Thought (b=1)':<25} {cot_success_rate:>9.0%} {cot_avg_error:>12.2f}")
    print(f"  {'Random (50)':<25} {random_success_rate:>9.0%} {random_avg_error:>12.2f}")
    print("-" * 60)

    # 
    print("\n:")
    print("  1. ToT  + ，")
    print("  2. CoT (breadth=1) ，")
    print("  3. ，")
    print("  4. (Test-Time Search)：")
    print("     ")
    print("  5. ToT  LLM ")

    # ----  ----
    results = {
        'tot': {
            'success_rate': tot_success_rate,
            'avg_error': tot_avg_error,
            'breadth_success_rates': breadth_success_rates,
        },
        'cot': {
            'success_rate': cot_success_rate,
            'avg_error': cot_avg_error,
        },
        'random': {
            'success_rate': random_success_rate,
            'avg_error': random_avg_error,
        },
    }

    # 1: 
    if first_tree_data:
        fig1 = visualize_search_tree(first_tree_data,
                                     title="Tree of Thought （1）")
        fig1.savefig('output/tot_search_tree.png', dpi=150, bbox_inches='tight')
        print("\n: output/tot_search_tree.png")

    # 2: 
    problem_labels = [str(p) for p in problems]
    fig2 = visualize_comparison(results, problem_labels)
    fig2.savefig('output/tot_comparison.png', dpi=150, bbox_inches='tight')
    print(": output/tot_comparison.png")

    plt.show()


# ==========================================
# ：
# ==========================================
if __name__ == "__main__":
    # ，
    np.random.seed(42)

    # 
    run_experiment()
