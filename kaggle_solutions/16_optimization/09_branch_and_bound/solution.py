"""
Branch and Bound Algorithm Implementation
=========================================

This solution implements comprehensive branch and bound algorithms for
integer and mixed-integer programming problems.

Mathematical Background:
-----------------------
Branch and Bound is a divide-and-conquer approach for discrete optimization:
1. Branch: Partition the solution space into smaller subproblems
2. Bound: Compute bounds to prune subproblems that cannot contain optimal solution
3. Explore: Systematically explore the search tree using various strategies

Common applications:
- Integer Linear Programming (ILP)
- Traveling Salesman Problem (TSP)
- Job Scheduling
- Knapsack Problems

Author: Optimization Expert
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from queue import PriorityQueue
import time
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Node:
    """Represents a node in the branch and bound tree."""
    level: int
    bound: float
    value: float
    solution: List[float]
    fixed_variables: Dict[int, int]

    def __lt__(self, other):
        """Comparison for priority queue (higher bound = higher priority)."""
        return self.bound > other.bound


class BranchAndBound:
    """
    Branch and Bound solver for Integer Linear Programming.
    """

    def __init__(self, c: np.ndarray, A: np.ndarray, b: np.ndarray,
                 integer_vars: Optional[List[int]] = None):
        """
        Initialize Branch and Bound solver.

        Args:
            c: Objective coefficients (maximize c^T x)
            A: Constraint matrix (Ax <= b)
            b: Right-hand side
            integer_vars: Indices of integer variables (None = all integer)
        """
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.n = len(c)
        self.integer_vars = integer_vars if integer_vars else list(range(self.n))

        self.best_solution = None
        self.best_value = -np.inf
        self.nodes_explored = 0
        self.nodes_pruned = 0
        self.tree_history = []

    def solve(self, strategy='best_first', max_nodes=10000) -> Dict:
        """
        Solve ILP using branch and bound.

        Args:
            strategy: 'best_first', 'depth_first', or 'breadth_first'
            max_nodes: Maximum nodes to explore

        Returns:
            Dictionary with solution details
        """
        start_time = time.time()

        # Solve LP relaxation for root node
        root_solution, root_value = self._solve_lp_relaxation({})

        if root_solution is None:
            return {'status': 'infeasible'}

        # Initialize with root node
        root_node = Node(
            level=0,
            bound=root_value,
            value=root_value,
            solution=root_solution,
            fixed_variables={}
        )

        if strategy == 'best_first':
            queue = PriorityQueue()
            queue.put((0, root_node))
        elif strategy == 'depth_first':
            queue = [root_node]
        else:  # breadth_first
            queue = [root_node]

        while (not self._is_queue_empty(queue, strategy) and
               self.nodes_explored < max_nodes):

            # Get next node based on strategy
            node = self._get_next_node(queue, strategy)
            self.nodes_explored += 1

            # Record tree exploration
            self.tree_history.append({
                'node': self.nodes_explored,
                'level': node.level,
                'bound': node.bound,
                'best_value': self.best_value
            })

            # Prune if bound is worse than current best
            if node.bound <= self.best_value:
                self.nodes_pruned += 1
                continue

            # Check if solution is integer
            if self._is_integer_solution(node.solution):
                if node.value > self.best_value:
                    self.best_value = node.value
                    self.best_solution = node.solution.copy()
                continue

            # Branch on fractional variable
            branch_var = self._select_branching_variable(node.solution)

            if branch_var is None:
                continue

            # Create child nodes
            for branch_value in [np.floor(node.solution[branch_var]),
                                np.ceil(node.solution[branch_var])]:

                new_fixed = node.fixed_variables.copy()
                new_fixed[branch_var] = int(branch_value)

                # Solve LP relaxation with fixed variables
                child_solution, child_value = self._solve_lp_relaxation(new_fixed)

                if child_solution is not None and child_value > self.best_value:
                    child_node = Node(
                        level=node.level + 1,
                        bound=child_value,
                        value=child_value,
                        solution=child_solution,
                        fixed_variables=new_fixed
                    )

                    if strategy == 'best_first':
                        queue.put((self.nodes_explored, child_node))
                    elif strategy == 'depth_first':
                        queue.append(child_node)
                    else:  # breadth_first
                        queue.insert(0, child_node)

        elapsed_time = time.time() - start_time

        return {
            'status': 'optimal' if self.best_solution is not None else 'no_solution',
            'solution': self.best_solution,
            'objective': self.best_value,
            'nodes_explored': self.nodes_explored,
            'nodes_pruned': self.nodes_pruned,
            'time': elapsed_time,
            'tree_history': self.tree_history
        }

    def _solve_lp_relaxation(self, fixed_vars: Dict[int, int]) -> Tuple[Optional[np.ndarray], float]:
        """Solve LP relaxation with some variables fixed."""
        from scipy.optimize import linprog

        # Create bounds: 0 <= x_i <= infinity (or fixed value)
        bounds = []
        for i in range(self.n):
            if i in fixed_vars:
                bounds.append((fixed_vars[i], fixed_vars[i]))
            else:
                bounds.append((0, None))

        # Solve LP (scipy minimizes, so negate objective)
        result = linprog(-self.c, A_ub=self.A, b_ub=self.b,
                        bounds=bounds, method='highs')

        if result.success:
            return result.x, -result.fun
        else:
            return None, -np.inf

    def _is_integer_solution(self, solution: np.ndarray, tol=1e-6) -> bool:
        """Check if solution is integer for integer variables."""
        for i in self.integer_vars:
            if abs(solution[i] - round(solution[i])) > tol:
                return False
        return True

    def _select_branching_variable(self, solution: np.ndarray) -> Optional[int]:
        """Select variable to branch on (most fractional)."""
        max_frac = 0
        branch_var = None

        for i in self.integer_vars:
            frac = abs(solution[i] - round(solution[i]))
            if frac > max_frac:
                max_frac = frac
                branch_var = i

        return branch_var if max_frac > 1e-6 else None

    def _is_queue_empty(self, queue, strategy: str) -> bool:
        """Check if queue is empty."""
        if strategy == 'best_first':
            return queue.empty()
        else:
            return len(queue) == 0

    def _get_next_node(self, queue, strategy: str) -> Node:
        """Get next node from queue based on strategy."""
        if strategy == 'best_first':
            _, node = queue.get()
            return node
        elif strategy == 'depth_first':
            return queue.pop()
        else:  # breadth_first
            return queue.pop(0)


class KnapsackBnB:
    """Branch and Bound for 0-1 Knapsack Problem."""

    def __init__(self, weights: np.ndarray, values: np.ndarray, capacity: float):
        self.weights = weights
        self.values = values
        self.capacity = capacity
        self.n = len(weights)
        self.best_value = 0
        self.best_solution = np.zeros(self.n, dtype=int)
        self.nodes_explored = 0

    def solve(self) -> Dict:
        """Solve knapsack using branch and bound."""
        # Sort items by value/weight ratio
        ratios = self.values / self.weights
        indices = np.argsort(ratios)[::-1]

        self._branch(0, 0, 0, np.zeros(self.n, dtype=int), indices)

        return {
            'solution': self.best_solution,
            'value': self.best_value,
            'nodes_explored': self.nodes_explored
        }

    def _branch(self, level: int, current_weight: float, current_value: float,
                current_solution: np.ndarray, indices: np.ndarray):
        """Recursive branching function."""
        self.nodes_explored += 1

        if level == self.n:
            if current_value > self.best_value:
                self.best_value = current_value
                self.best_solution = current_solution.copy()
            return

        idx = indices[level]

        # Compute upper bound using fractional knapsack
        bound = self._compute_bound(level, current_weight, current_value, indices)

        if bound <= self.best_value:
            return  # Prune

        # Branch 1: Include item
        if current_weight + self.weights[idx] <= self.capacity:
            current_solution[idx] = 1
            self._branch(level + 1,
                        current_weight + self.weights[idx],
                        current_value + self.values[idx],
                        current_solution, indices)
            current_solution[idx] = 0

        # Branch 2: Exclude item
        self._branch(level + 1, current_weight, current_value,
                    current_solution, indices)

    def _compute_bound(self, level: int, current_weight: float,
                       current_value: float, indices: np.ndarray) -> float:
        """Compute upper bound using fractional knapsack."""
        bound = current_value
        remaining_capacity = self.capacity - current_weight

        for i in range(level, self.n):
            idx = indices[i]
            if self.weights[idx] <= remaining_capacity:
                bound += self.values[idx]
                remaining_capacity -= self.weights[idx]
            else:
                bound += self.values[idx] * (remaining_capacity / self.weights[idx])
                break

        return bound


def benchmark_strategies():
    """Benchmark different branching strategies."""
    np.random.seed(42)

    results = []
    problem_sizes = [5, 8, 10]

    for n in problem_sizes:
        # Generate random ILP problem
        c = np.random.randint(1, 10, n)
        A = np.random.randint(1, 5, (n//2, n))
        b = np.random.randint(5, 15, n//2)

        for strategy in ['best_first', 'depth_first', 'breadth_first']:
            solver = BranchAndBound(c, A, b)
            result = solver.solve(strategy=strategy, max_nodes=1000)

            results.append({
                'size': n,
                'strategy': strategy,
                'objective': result['objective'],
                'nodes_explored': result['nodes_explored'],
                'nodes_pruned': result['nodes_pruned'],
                'time': result['time']
            })

    return pd.DataFrame(results)


def visualize_branch_and_bound_tree():
    """Visualize branch and bound exploration."""
    # Simple problem for visualization
    c = np.array([5, 4, 3])
    A = np.array([[2, 3, 1]])
    b = np.array([10])

    solver = BranchAndBound(c, A, b)
    result = solver.solve(strategy='best_first', max_nodes=50)

    # Extract tree history
    history_df = pd.DataFrame(result['tree_history'])

    # Plot exploration over time
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Subplot 1: Nodes explored vs bound
    axes[0, 0].plot(history_df['node'], history_df['bound'], 'b-', alpha=0.6, label='Node Bound')
    axes[0, 0].plot(history_df['node'], history_df['best_value'], 'r-', linewidth=2, label='Best Value')
    axes[0, 0].set_xlabel('Nodes Explored', fontsize=12)
    axes[0, 0].set_ylabel('Objective Value', fontsize=12)
    axes[0, 0].set_title('Branch and Bound Convergence', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Subplot 2: Tree depth distribution
    depth_counts = history_df['level'].value_counts().sort_index()
    axes[0, 1].bar(depth_counts.index, depth_counts.values, color='steelblue', alpha=0.7)
    axes[0, 1].set_xlabel('Tree Level', fontsize=12)
    axes[0, 1].set_ylabel('Number of Nodes', fontsize=12)
    axes[0, 1].set_title('Tree Depth Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Subplot 3: Bound values over time
    axes[1, 0].scatter(history_df['node'], history_df['bound'],
                       c=history_df['level'], cmap='viridis', alpha=0.6)
    axes[1, 0].set_xlabel('Nodes Explored', fontsize=12)
    axes[1, 0].set_ylabel('Bound Value', fontsize=12)
    axes[1, 0].set_title('Bound Values (colored by level)', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Subplot 4: Summary statistics
    axes[1, 1].axis('off')
    summary_text = f"""Branch and Bound Summary

    Total Nodes Explored: {result['nodes_explored']}
    Nodes Pruned: {result['nodes_pruned']}
    Pruning Rate: {result['nodes_pruned']/result['nodes_explored']*100:.1f}%

    Optimal Value: {result['objective']:.2f}
    Solution: {result['solution']}

    Execution Time: {result['time']:.4f} seconds
    Max Tree Depth: {history_df['level'].max()}
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                    verticalalignment='center')

    plt.tight_layout()
    plt.savefig('branch_and_bound_tree.png', dpi=300, bbox_inches='tight')
    plt.close()


def compare_with_exhaustive_search():
    """Compare B&B with exhaustive search."""
    np.random.seed(42)

    sizes = range(4, 12)
    results = {'size': [], 'bnb_nodes': [], 'exhaustive_nodes': [], 'speedup': []}

    for n in sizes:
        # Simple knapsack problem
        weights = np.random.randint(1, 10, n)
        values = np.random.randint(1, 20, n)
        capacity = np.sum(weights) // 2

        # Branch and Bound
        bnb_solver = KnapsackBnB(weights, values, capacity)
        bnb_result = bnb_solver.solve()

        # Exhaustive search nodes = 2^n
        exhaustive_nodes = 2 ** n

        results['size'].append(n)
        results['bnb_nodes'].append(bnb_result['nodes_explored'])
        results['exhaustive_nodes'].append(exhaustive_nodes)
        results['speedup'].append(exhaustive_nodes / bnb_result['nodes_explored'])

    df = pd.DataFrame(results)

    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Nodes explored
    axes[0].semilogy(df['size'], df['bnb_nodes'], 'o-', label='Branch & Bound', linewidth=2)
    axes[0].semilogy(df['size'], df['exhaustive_nodes'], 's-', label='Exhaustive Search', linewidth=2)
    axes[0].set_xlabel('Problem Size', fontsize=12)
    axes[0].set_ylabel('Nodes Explored (log scale)', fontsize=12)
    axes[0].set_title('Nodes Explored Comparison', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Speedup factor
    axes[1].plot(df['size'], df['speedup'], 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Problem Size', fontsize=12)
    axes[1].set_ylabel('Speedup Factor', fontsize=12)
    axes[1].set_title('Branch & Bound Speedup', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bnb_vs_exhaustive.png', dpi=300, bbox_inches='tight')
    plt.close()

    return df


def main():
    """Main execution function."""
    print("="*70)
    print("Branch and Bound Algorithm Implementation")
    print("="*70)

    # Example 1: Integer Linear Programming
    print("\n1. Integer Linear Programming")
    print("-" * 70)
    c = np.array([5, 4, 3])
    A = np.array([[2, 3, 1], [1, 2, 3]])
    b = np.array([10, 12])

    solver = BranchAndBound(c, A, b)
    result = solver.solve(strategy='best_first')

    print(f"Status: {result['status']}")
    print(f"Optimal solution: {result['solution']}")
    print(f"Optimal value: {result['objective']:.2f}")
    print(f"Nodes explored: {result['nodes_explored']}")
    print(f"Nodes pruned: {result['nodes_pruned']}")
    print(f"Time: {result['time']:.4f} seconds")

    # Example 2: Knapsack Problem
    print("\n2. 0-1 Knapsack Problem")
    print("-" * 70)
    weights = np.array([2, 3, 4, 5])
    values = np.array([3, 4, 5, 6])
    capacity = 8

    knapsack = KnapsackBnB(weights, values, capacity)
    knapsack_result = knapsack.solve()

    print(f"Items selected: {knapsack_result['solution']}")
    print(f"Total value: {knapsack_result['value']}")
    print(f"Nodes explored: {knapsack_result['nodes_explored']}")

    # Example 3: Strategy Comparison
    print("\n3. Branching Strategy Comparison")
    print("-" * 70)
    benchmark_df = benchmark_strategies()
    print(benchmark_df.to_string(index=False))

    # Visualize strategy comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for strategy in ['best_first', 'depth_first', 'breadth_first']:
        data = benchmark_df[benchmark_df['strategy'] == strategy]
        axes[0].plot(data['size'], data['nodes_explored'], 'o-', label=strategy, linewidth=2)
        axes[1].plot(data['size'], data['time'], 's-', label=strategy, linewidth=2)

    axes[0].set_xlabel('Problem Size', fontsize=12)
    axes[0].set_ylabel('Nodes Explored', fontsize=12)
    axes[0].set_title('Nodes Explored by Strategy', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Problem Size', fontsize=12)
    axes[1].set_ylabel('Time (seconds)', fontsize=12)
    axes[1].set_title('Execution Time by Strategy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('strategy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Example 4: Tree visualization
    print("\n4. Branch and Bound Tree Visualization")
    print("-" * 70)
    visualize_branch_and_bound_tree()
    print("Tree visualization saved to 'branch_and_bound_tree.png'")

    # Example 5: Comparison with exhaustive search
    print("\n5. Comparison with Exhaustive Search")
    print("-" * 70)
    comparison_df = compare_with_exhaustive_search()
    print(comparison_df.to_string(index=False))
    print("Comparison plot saved to 'bnb_vs_exhaustive.png'")

    print("\n" + "="*70)
    print("Analysis complete! Generated visualizations:")
    print("  - branch_and_bound_tree.png")
    print("  - strategy_comparison.png")
    print("  - bnb_vs_exhaustive.png")
    print("="*70)


if __name__ == "__main__":
    main()
