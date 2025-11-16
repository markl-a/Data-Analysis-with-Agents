"""
Knapsack Problem Solutions
===========================

This example demonstrates multiple approaches to solving the Knapsack Problem,
a classic optimization problem in computer science and operations research.

Problem: Given a set of items with weights and values, select items to maximize
total value while staying within a weight capacity constraint.

Author: Kaggle Solutions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import time
import warnings
warnings.filterwarnings('ignore')


class KnapsackSolver:
    """
    Comprehensive Knapsack solver with multiple algorithms.
    """

    def __init__(self, seed=42):
        """
        Initialize the Knapsack solver.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        self.results = {}

    def generate_problem(self, n_items=20, capacity=100) -> Dict:
        """
        Generate a random knapsack problem instance.

        Args:
            n_items: Number of items
            capacity: Knapsack capacity

        Returns:
            Dictionary with problem data
        """
        # Generate random weights and values
        weights = np.random.randint(5, 25, n_items)
        values = np.random.randint(10, 100, n_items)

        # Calculate value-to-weight ratio
        ratios = values / weights

        problem = {
            'n_items': n_items,
            'capacity': capacity,
            'weights': weights,
            'values': values,
            'ratios': ratios,
            'item_names': [f'Item {i+1}' for i in range(n_items)]
        }

        return problem

    def solve_dynamic_programming(self, problem: Dict) -> Dict:
        """
        Solve 0/1 Knapsack using Dynamic Programming (optimal).

        Args:
            problem: Problem definition dictionary

        Returns:
            Dictionary with solution results
        """
        print("\n" + "="*60)
        print("Method 1: Dynamic Programming (Optimal)")
        print("="*60)

        start_time = time.time()

        n = problem['n_items']
        W = problem['capacity']
        weights = problem['weights']
        values = problem['values']

        # Create DP table
        # dp[i][w] = maximum value achievable with first i items and capacity w
        dp = np.zeros((n + 1, W + 1), dtype=int)

        print("\nBuilding DP table...")

        # Fill DP table
        for i in range(1, n + 1):
            for w in range(W + 1):
                # Option 1: Don't include item i-1
                dp[i][w] = dp[i-1][w]

                # Option 2: Include item i-1 (if it fits)
                if weights[i-1] <= w:
                    dp[i][w] = max(dp[i][w],
                                  dp[i-1][w - weights[i-1]] + values[i-1])

        # Backtrack to find selected items
        selected = np.zeros(n, dtype=bool)
        w = W
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i-1][w]:
                selected[i-1] = True
                w -= weights[i-1]

        total_value = dp[n][W]
        total_weight = np.sum(weights[selected])
        elapsed_time = time.time() - start_time

        solution = {
            'method': 'Dynamic Programming',
            'selected': selected,
            'total_value': total_value,
            'total_weight': total_weight,
            'n_selected': np.sum(selected),
            'success': True,
            'time': elapsed_time
        }

        print(f"Optimal value: {total_value}")
        print(f"Total weight: {total_weight}/{W}")
        print(f"Items selected: {np.sum(selected)}/{n}")
        print(f"Time: {elapsed_time:.4f} seconds")

        return solution

    def solve_greedy_value(self, problem: Dict) -> Dict:
        """
        Solve using greedy heuristic: select by highest value.

        Args:
            problem: Problem definition dictionary

        Returns:
            Dictionary with solution results
        """
        print("\n" + "="*60)
        print("Method 2: Greedy by Value")
        print("="*60)

        start_time = time.time()

        n = problem['n_items']
        W = problem['capacity']
        weights = problem['weights']
        values = problem['values']

        # Sort by value (descending)
        order = np.argsort(values)[::-1]

        selected = np.zeros(n, dtype=bool)
        current_weight = 0

        print("\nGreedy selection:")
        for idx in order:
            if current_weight + weights[idx] <= W:
                selected[idx] = True
                current_weight += weights[idx]
                print(f"  Selected {problem['item_names'][idx]}: "
                      f"value={values[idx]}, weight={weights[idx]}")

        total_value = np.sum(values[selected])
        total_weight = current_weight
        elapsed_time = time.time() - start_time

        solution = {
            'method': 'Greedy by Value',
            'selected': selected,
            'total_value': total_value,
            'total_weight': total_weight,
            'n_selected': np.sum(selected),
            'success': True,
            'time': elapsed_time
        }

        print(f"\nTotal value: {total_value}")
        print(f"Total weight: {total_weight}/{W}")
        print(f"Time: {elapsed_time:.4f} seconds")

        return solution

    def solve_greedy_ratio(self, problem: Dict) -> Dict:
        """
        Solve using greedy heuristic: select by highest value/weight ratio.

        Args:
            problem: Problem definition dictionary

        Returns:
            Dictionary with solution results
        """
        print("\n" + "="*60)
        print("Method 3: Greedy by Value/Weight Ratio")
        print("="*60)

        start_time = time.time()

        n = problem['n_items']
        W = problem['capacity']
        weights = problem['weights']
        values = problem['values']
        ratios = problem['ratios']

        # Sort by ratio (descending)
        order = np.argsort(ratios)[::-1]

        selected = np.zeros(n, dtype=bool)
        current_weight = 0

        print("\nGreedy selection:")
        for idx in order:
            if current_weight + weights[idx] <= W:
                selected[idx] = True
                current_weight += weights[idx]
                print(f"  Selected {problem['item_names'][idx]}: "
                      f"ratio={ratios[idx]:.2f}, value={values[idx]}, weight={weights[idx]}")

        total_value = np.sum(values[selected])
        total_weight = current_weight
        elapsed_time = time.time() - start_time

        solution = {
            'method': 'Greedy by Ratio',
            'selected': selected,
            'total_value': total_value,
            'total_weight': total_weight,
            'n_selected': np.sum(selected),
            'success': True,
            'time': elapsed_time
        }

        print(f"\nTotal value: {total_value}")
        print(f"Total weight: {total_weight}/{W}")
        print(f"Time: {elapsed_time:.4f} seconds")

        return solution

    def solve_branch_and_bound(self, problem: Dict) -> Dict:
        """
        Solve using Branch and Bound algorithm.

        Args:
            problem: Problem definition dictionary

        Returns:
            Dictionary with solution results
        """
        print("\n" + "="*60)
        print("Method 4: Branch and Bound")
        print("="*60)

        start_time = time.time()

        n = problem['n_items']
        W = problem['capacity']
        weights = problem['weights']
        values = problem['values']
        ratios = problem['ratios']

        # Sort items by ratio (for better bounds)
        order = np.argsort(ratios)[::-1]
        weights_sorted = weights[order]
        values_sorted = values[order]

        best_value = 0
        best_solution = np.zeros(n, dtype=bool)
        nodes_explored = [0]  # Use list to allow modification in nested function

        def fractional_bound(index, current_weight, current_value):
            """Calculate upper bound using fractional knapsack."""
            if current_weight >= W:
                return 0

            bound = current_value
            total_weight = current_weight

            for j in range(index, n):
                if total_weight + weights_sorted[j] <= W:
                    total_weight += weights_sorted[j]
                    bound += values_sorted[j]
                else:
                    # Add fraction of remaining item
                    bound += (W - total_weight) * ratios[order[j]]
                    break

            return bound

        def branch_and_bound_recursive(index, current_weight, current_value, current_solution):
            """Recursive branch and bound."""
            nonlocal best_value, best_solution
            nodes_explored[0] += 1

            if index == n or current_weight == W:
                if current_value > best_value:
                    best_value = current_value
                    best_solution = current_solution.copy()
                return

            # Calculate bound
            bound = fractional_bound(index, current_weight, current_value)

            if bound <= best_value:
                return  # Prune this branch

            # Branch 1: Include current item
            if current_weight + weights_sorted[index] <= W:
                current_solution[order[index]] = True
                branch_and_bound_recursive(
                    index + 1,
                    current_weight + weights_sorted[index],
                    current_value + values_sorted[index],
                    current_solution
                )
                current_solution[order[index]] = False

            # Branch 2: Exclude current item
            branch_and_bound_recursive(index, current_weight, current_value, current_solution)

        # Start recursion
        print("\nExploring search tree...")
        branch_and_bound_recursive(0, 0, 0, np.zeros(n, dtype=bool))

        total_weight = np.sum(weights[best_solution])
        elapsed_time = time.time() - start_time

        solution = {
            'method': 'Branch and Bound',
            'selected': best_solution,
            'total_value': best_value,
            'total_weight': total_weight,
            'n_selected': np.sum(best_solution),
            'success': True,
            'time': elapsed_time,
            'nodes_explored': nodes_explored[0]
        }

        print(f"\nNodes explored: {nodes_explored[0]}")
        print(f"Optimal value: {best_value}")
        print(f"Total weight: {total_weight}/{W}")
        print(f"Time: {elapsed_time:.4f} seconds")

        return solution

    def solve_all_methods(self, problem: Dict):
        """Solve the problem using all available methods."""
        print("\nSOLVING KNAPSACK PROBLEM")
        print("="*60)
        print(f"Number of items: {problem['n_items']}")
        print(f"Knapsack capacity: {problem['capacity']}")
        print(f"Total items weight: {np.sum(problem['weights'])}")
        print(f"Total items value: {np.sum(problem['values'])}")
        print("="*60)

        # Solve with different methods
        self.results['dp'] = self.solve_dynamic_programming(problem)
        self.results['greedy_value'] = self.solve_greedy_value(problem)
        self.results['greedy_ratio'] = self.solve_greedy_ratio(problem)

        # Branch and bound (only for smaller instances)
        if problem['n_items'] <= 25:
            self.results['bb'] = self.solve_branch_and_bound(problem)
        else:
            print("\n" + "="*60)
            print("Skipping Branch and Bound (too many items)")
            print("="*60)

        return self.results

    def visualize_solution(self, problem: Dict, solution: Dict):
        """
        Visualize a single solution.

        Args:
            problem: Problem definition dictionary
            solution: Solution dictionary
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        selected = solution['selected']
        weights = problem['weights']
        values = problem['values']
        ratios = problem['ratios']

        # Plot 1: Selected vs Not Selected items
        ax = axes[0, 0]

        selected_idx = np.where(selected)[0]
        not_selected_idx = np.where(~selected)[0]

        ax.scatter(weights[not_selected_idx], values[not_selected_idx],
                  s=200, c='lightgray', marker='o', label='Not Selected',
                  edgecolors='black', linewidths=1, alpha=0.6)
        ax.scatter(weights[selected_idx], values[selected_idx],
                  s=300, c='green', marker='o', label='Selected',
                  edgecolors='black', linewidths=2)

        # Add item labels
        for i in selected_idx:
            ax.annotate(f'{i+1}', xy=(weights[i], values[i]),
                       fontsize=9, ha='center', va='center',
                       color='white', fontweight='bold')

        ax.set_xlabel('Weight', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(f'{solution["method"]}: Item Selection\nValue={solution["total_value"]}, Weight={solution["total_weight"]}/{problem["capacity"]}',
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Capacity utilization
        ax = axes[0, 1]

        used = solution['total_weight']
        unused = problem['capacity'] - used
        utilization = (used / problem['capacity']) * 100

        wedges, texts, autotexts = ax.pie([used, unused],
                                          labels=['Used', 'Unused'],
                                          autopct='%1.1f%%',
                                          colors=['#2ecc71', '#ecf0f1'],
                                          startangle=90,
                                          textprops={'fontsize': 12, 'fontweight': 'bold'})

        ax.set_title(f'Capacity Utilization\n{used}/{problem["capacity"]} ({utilization:.1f}%)',
                    fontsize=14, fontweight='bold')

        # Plot 3: Item characteristics
        ax = axes[1, 0]

        x = np.arange(problem['n_items'])
        width = 0.35

        bars1 = ax.bar(x - width/2, weights, width, label='Weight',
                      color='blue', alpha=0.7)
        bars2 = ax.bar(x + width/2, values, width, label='Value',
                      color='orange', alpha=0.7)

        # Highlight selected items
        for i in selected_idx:
            bars1[i].set_edgecolor('red')
            bars1[i].set_linewidth(3)
            bars2[i].set_edgecolor('red')
            bars2[i].set_linewidth(3)

        ax.set_xlabel('Item', fontsize=12)
        ax.set_ylabel('Weight / Value', fontsize=12)
        ax.set_title('Item Characteristics (Selected items have red border)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 4: Value/Weight ratios
        ax = axes[1, 1]

        colors = ['green' if s else 'lightgray' for s in selected]
        bars = ax.barh(range(problem['n_items']), ratios, color=colors,
                      edgecolor='black', linewidth=1, alpha=0.7)

        ax.set_yticks(range(problem['n_items']))
        ax.set_yticklabels([f'Item {i+1}' for i in range(problem['n_items'])])
        ax.set_xlabel('Value/Weight Ratio', fontsize=12)
        ax.set_title('Item Efficiency (Green = Selected)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()

        plt.tight_layout()
        plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/16_optimization/04_knapsack/knapsack_solution.png',
                    dpi=300, bbox_inches='tight')
        print("\nSolution visualization saved to: knapsack_solution.png")
        plt.show()

    def visualize_comparison(self, problem: Dict):
        """
        Compare all solution methods.

        Args:
            problem: Problem definition dictionary
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        methods = [v['method'] for v in self.results.values()]
        values = [v['total_value'] for v in self.results.values()]
        times = [v['time'] for v in self.results.values()]

        # Plot 1: Value comparison
        ax = axes[0]
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12'][:len(methods)]

        bars = ax.bar(methods, values, color=colors, alpha=0.7,
                     edgecolor='black', linewidth=2)

        # Highlight optimal
        optimal_value = max(values)
        for i, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

            if value == optimal_value:
                bar.set_edgecolor('gold')
                bar.set_linewidth(4)

        ax.set_ylabel('Total Value', fontsize=12)
        ax.set_title('Total Value Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 2: Time comparison
        ax = axes[1]

        bars = ax.bar(methods, times, color=colors, alpha=0.7,
                     edgecolor='black', linewidth=2)

        for bar, t in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{t:.4f}s',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_ylabel('Computation Time (seconds)', fontsize=12)
        ax.set_title('Computation Time Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/16_optimization/04_knapsack/knapsack_comparison.png',
                    dpi=300, bbox_inches='tight')
        print("Comparison visualization saved to: knapsack_comparison.png")
        plt.show()


def main():
    """Main execution function."""
    print("="*60)
    print("KNAPSACK PROBLEM OPTIMIZATION")
    print("="*60)

    # Create solver
    solver = KnapsackSolver(seed=42)

    # Generate problem
    problem = solver.generate_problem(n_items=20, capacity=100)

    # Solve using all methods
    results = solver.solve_all_methods(problem)

    # Compare results
    print("\n" + "="*60)
    print("Comparison of Methods")
    print("="*60)

    comparison_data = []
    for method_name, result in results.items():
        comparison_data.append({
            'Method': result['method'],
            'Total Value': result['total_value'],
            'Total Weight': f"{result['total_weight']}/{problem['capacity']}",
            'Items Selected': f"{result['n_selected']}/{problem['n_items']}",
            'Time (s)': f"{result['time']:.4f}"
        })

    df_comparison = pd.DataFrame(comparison_data)
    print("\n", df_comparison.to_string(index=False))

    # Find optimal solution
    optimal_value = max(r['total_value'] for r in results.values())
    optimal_methods = [r['method'] for r in results.values() if r['total_value'] == optimal_value]
    print(f"\nOptimal value: {optimal_value}")
    print(f"Optimal method(s): {', '.join(optimal_methods)}")

    # Visualize best solution
    best_result = results['dp']
    solver.visualize_solution(problem, best_result)

    # Compare all methods
    solver.visualize_comparison(problem)

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
