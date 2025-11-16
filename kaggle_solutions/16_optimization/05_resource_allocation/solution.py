"""
Resource Allocation Optimization
=================================

This example demonstrates resource allocation optimization for multi-project,
multi-resource scenarios commonly found in organizations.

Problem: Allocate limited resources (people, budget, equipment) across multiple
projects to maximize overall value while respecting resource constraints and
project requirements.

Author: Kaggle Solutions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog, minimize
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class ResourceAllocationOptimizer:
    """
    Multi-resource allocation optimizer with various objectives.
    """

    def __init__(self, seed=42):
        """Initialize the optimizer."""
        self.seed = seed
        np.random.seed(seed)
        self.results = {}

    def generate_problem(self, n_projects=8, n_resources=4) -> Dict:
        """
        Generate a resource allocation problem.

        Args:
            n_projects: Number of projects
            n_resources: Number of resource types

        Returns:
            Problem dictionary
        """
        # Project values (expected ROI)
        values = np.random.randint(50, 200, n_projects)

        # Resource requirements (each project needs resources)
        # requirements[i][j] = units of resource j needed by project i
        requirements = np.random.randint(5, 20, (n_projects, n_resources))

        # Available resources
        available = np.random.randint(30, 50, n_resources)

        # Project priorities (1-5 scale)
        priorities = np.random.randint(1, 6, n_projects)

        # Minimum allocation thresholds (0-1, fraction of requirements)
        min_allocation = np.random.uniform(0.3, 0.5, n_projects)

        problem = {
            'n_projects': n_projects,
            'n_resources': n_resources,
            'values': values,
            'requirements': requirements,
            'available': available,
            'priorities': priorities,
            'min_allocation': min_allocation,
            'project_names': [f'Project {chr(65+i)}' for i in range(n_projects)],
            'resource_names': ['People', 'Budget', 'Equipment', 'Time'][:n_resources]
        }

        return problem

    def solve_proportional_allocation(self, problem: Dict) -> Dict:
        """
        Solve using proportional allocation based on project value.

        Args:
            problem: Problem dictionary

        Returns:
            Solution dictionary
        """
        print("\n" + "="*60)
        print("Method 1: Proportional Allocation")
        print("="*60)

        n_proj = problem['n_projects']
        n_res = problem['n_resources']
        values = problem['values']
        requirements = problem['requirements']
        available = problem['available']

        # Allocate proportionally to value
        total_value = np.sum(values)
        allocation_fractions = values / total_value

        # Calculate actual allocations
        allocations = np.zeros((n_proj, n_res))

        print("\nAllocation process:")
        for i in range(n_proj):
            for j in range(n_res):
                # Allocate based on fraction of total value
                desired = requirements[i][j] * allocation_fractions[i] * n_proj
                allocations[i][j] = min(desired, requirements[i][j])

        # Adjust to respect constraints
        for j in range(n_res):
            total_allocated = np.sum(allocations[:, j])
            if total_allocated > available[j]:
                # Scale down proportionally
                scale_factor = available[j] / total_allocated
                allocations[:, j] *= scale_factor
                print(f"  {problem['resource_names'][j]}: Scaled down by {scale_factor:.2f}")

        # Calculate achievement (fraction of requirements met)
        achievement = np.zeros(n_proj)
        for i in range(n_proj):
            achievement[i] = np.min(allocations[i] / requirements[i])

        total_value_achieved = np.sum(achievement * values)

        solution = {
            'method': 'Proportional Allocation',
            'allocations': allocations,
            'achievement': achievement,
            'total_value': total_value_achieved,
            'success': True
        }

        print(f"\nTotal value achieved: {total_value_achieved:.2f}")
        print(f"Average achievement: {np.mean(achievement)*100:.1f}%")

        return solution

    def solve_linear_programming(self, problem: Dict) -> Dict:
        """
        Solve using linear programming optimization.

        Args:
            problem: Problem dictionary

        Returns:
            Solution dictionary
        """
        print("\n" + "="*60)
        print("Method 2: Linear Programming Optimization")
        print("="*60)

        n_proj = problem['n_projects']
        n_res = problem['n_resources']
        values = problem['values']
        requirements = problem['requirements']
        available = problem['available']

        # Decision variables: allocation fraction for each project (0 to 1)
        # Maximize: sum of value * allocation_fraction

        # Objective: maximize total value
        c = -values  # Negative for minimization

        # Constraints: resource constraints
        # For each resource j: sum_i (requirements[i][j] * x[i]) <= available[j]
        A_ub = requirements.T  # Transpose to get resource rows
        b_ub = available

        # Bounds: 0 <= x[i] <= 1
        bounds = [(0, 1) for _ in range(n_proj)]

        result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        if result.success:
            achievement = result.x

            # Calculate allocations
            allocations = achievement[:, np.newaxis] * requirements

            total_value_achieved = -result.fun

            solution = {
                'method': 'Linear Programming',
                'allocations': allocations,
                'achievement': achievement,
                'total_value': total_value_achieved,
                'success': True
            }

            print(f"Status: {result.message}")
            print(f"Total value achieved: {total_value_achieved:.2f}")
            print(f"Average achievement: {np.mean(achievement)*100:.1f}%")

        else:
            solution = {
                'method': 'Linear Programming',
                'success': False,
                'message': result.message
            }
            print(f"Failed: {result.message}")

        return solution

    def solve_priority_based(self, problem: Dict) -> Dict:
        """
        Solve using priority-based greedy allocation.

        Args:
            problem: Problem dictionary

        Returns:
            Solution dictionary
        """
        print("\n" + "="*60)
        print("Method 3: Priority-Based Greedy Allocation")
        print("="*60)

        n_proj = problem['n_projects']
        n_res = problem['n_resources']
        values = problem['values']
        requirements = problem['requirements']
        available = problem['available'].copy()
        priorities = problem['priorities']

        # Sort projects by priority (descending)
        order = np.argsort(-priorities)

        achievement = np.zeros(n_proj)
        allocations = np.zeros((n_proj, n_res))

        print("\nAllocation by priority:")
        for idx in order:
            # Try to fully allocate to this project
            can_allocate = True
            max_fraction = 1.0

            # Check each resource
            for j in range(n_res):
                needed = requirements[idx][j]
                if needed > available[j]:
                    max_fraction = min(max_fraction, available[j] / needed)
                    can_allocate = False

            # Allocate what we can
            achievement[idx] = max_fraction
            for j in range(n_res):
                allocations[idx][j] = max_fraction * requirements[idx][j]
                available[j] -= allocations[idx][j]

            print(f"  {problem['project_names'][idx]} (Priority {priorities[idx]}): "
                  f"{max_fraction*100:.1f}% allocated")

        total_value_achieved = np.sum(achievement * values)

        solution = {
            'method': 'Priority-Based',
            'allocations': allocations,
            'achievement': achievement,
            'total_value': total_value_achieved,
            'success': True
        }

        print(f"\nTotal value achieved: {total_value_achieved:.2f}")
        print(f"Average achievement: {np.mean(achievement)*100:.1f}%")

        return solution

    def solve_maxmin_fairness(self, problem: Dict) -> Dict:
        """
        Solve using max-min fairness objective.

        Args:
            problem: Problem dictionary

        Returns:
            Solution dictionary
        """
        print("\n" + "="*60)
        print("Method 4: Max-Min Fairness")
        print("="*60)

        n_proj = problem['n_projects']
        n_res = problem['n_resources']
        requirements = problem['requirements']
        available = problem['available']
        values = problem['values']

        # Iteratively allocate to maintain fairness
        achievement = np.zeros(n_proj)
        allocations = np.zeros((n_proj, n_res))
        remaining = available.copy()

        active = np.ones(n_proj, dtype=bool)
        iteration = 0

        print("\nIterative fair allocation:")
        while np.any(active):
            iteration += 1

            # Calculate max possible allocation for each active project
            max_alloc = np.ones(n_proj)
            for i in range(n_proj):
                if active[i]:
                    for j in range(n_res):
                        if requirements[i][j] > 0:
                            max_alloc[i] = min(max_alloc[i],
                                             remaining[j] / requirements[i][j])

            if np.max(max_alloc[active]) <= 0:
                break

            # Find minimum achievable allocation among active
            min_alloc = np.min(max_alloc[active])

            # Allocate this amount to all active projects
            for i in range(n_proj):
                if active[i]:
                    increase = min_alloc
                    achievement[i] += increase
                    for j in range(n_res):
                        amount = increase * requirements[i][j]
                        allocations[i][j] += amount
                        remaining[j] -= amount

                    # Deactivate if fully allocated
                    if achievement[i] >= 0.999:  # Allow for floating point errors
                        active[i] = False

            print(f"  Iteration {iteration}: Allocated {min_alloc*100:.1f}% to {np.sum(active)} projects")

        # Cap at 100%
        achievement = np.minimum(achievement, 1.0)

        total_value_achieved = np.sum(achievement * values)

        solution = {
            'method': 'Max-Min Fairness',
            'allocations': allocations,
            'achievement': achievement,
            'total_value': total_value_achieved,
            'success': True
        }

        print(f"\nTotal value achieved: {total_value_achieved:.2f}")
        print(f"Average achievement: {np.mean(achievement)*100:.1f}%")
        print(f"Min achievement: {np.min(achievement)*100:.1f}%")
        print(f"Max achievement: {np.max(achievement)*100:.1f}%")

        return solution

    def solve_all_methods(self, problem: Dict):
        """Solve using all methods."""
        print("\nRESOURCE ALLOCATION OPTIMIZATION")
        print("="*60)
        print(f"Projects: {problem['n_projects']}")
        print(f"Resources: {problem['n_resources']}")
        print(f"Total project value: {np.sum(problem['values'])}")
        print("\nAvailable resources:")
        for i, name in enumerate(problem['resource_names']):
            print(f"  {name}: {problem['available'][i]}")
        print("="*60)

        self.results['proportional'] = self.solve_proportional_allocation(problem)
        self.results['lp'] = self.solve_linear_programming(problem)
        self.results['priority'] = self.solve_priority_based(problem)
        self.results['maxmin'] = self.solve_maxmin_fairness(problem)

        return self.results

    def visualize_allocation(self, problem: Dict):
        """Visualize resource allocation across all methods."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        methods = ['proportional', 'lp', 'priority', 'maxmin']
        titles = ['Proportional', 'Linear Programming', 'Priority-Based', 'Max-Min Fairness']

        for idx, (method_key, title) in enumerate(zip(methods, titles)):
            ax = axes[idx // 2, idx % 2]
            result = self.results[method_key]

            if not result['success']:
                ax.text(0.5, 0.5, 'Failed', ha='center', va='center')
                ax.set_title(title)
                continue

            achievement = result['achievement']
            projects = problem['project_names']

            # Stacked bar chart of resource allocation
            allocations = result['allocations']

            x = np.arange(len(projects))
            width = 0.8

            bottom = np.zeros(len(projects))
            colors = plt.cm.Set3(np.linspace(0, 1, problem['n_resources']))

            for j in range(problem['n_resources']):
                ax.bar(x, allocations[:, j], width, bottom=bottom,
                      label=problem['resource_names'][j],
                      color=colors[j], alpha=0.8, edgecolor='black', linewidth=0.5)
                bottom += allocations[:, j]

            ax.set_xticks(x)
            ax.set_xticklabels(projects, rotation=45, ha='right')
            ax.set_ylabel('Resource Units', fontsize=10)
            ax.set_title(f'{title}\nTotal Value: {result["total_value"]:.2f}',
                        fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/16_optimization/05_resource_allocation/allocation_comparison.png',
                    dpi=300, bbox_inches='tight')
        print("\nAllocation visualization saved to: allocation_comparison.png")
        plt.show()

    def visualize_achievement(self, problem: Dict):
        """Visualize project achievement rates."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Achievement comparison
        ax = axes[0]

        methods = ['Proportional', 'Linear Programming', 'Priority-Based', 'Max-Min Fairness']
        method_keys = ['proportional', 'lp', 'priority', 'maxmin']

        x = np.arange(problem['n_projects'])
        width = 0.2

        for idx, (method, key) in enumerate(zip(methods, method_keys)):
            result = self.results[key]
            if result['success']:
                offset = (idx - 1.5) * width
                ax.bar(x + offset, result['achievement'] * 100, width,
                      label=method, alpha=0.8, edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Projects', fontsize=12)
        ax.set_ylabel('Achievement (%)', fontsize=12)
        ax.set_title('Project Achievement by Method', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(problem['project_names'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=100, color='r', linestyle='--', alpha=0.5)

        # Plot 2: Total value comparison
        ax = axes[1]

        total_values = [self.results[key]['total_value'] for key in method_keys
                       if self.results[key]['success']]
        valid_methods = [m for m, k in zip(methods, method_keys)
                        if self.results[k]['success']]

        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        bars = ax.bar(valid_methods, total_values, color=colors[:len(valid_methods)],
                     alpha=0.7, edgecolor='black', linewidth=2)

        # Highlight best
        best_idx = np.argmax(total_values)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(4)

        for bar, val in zip(bars, total_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_ylabel('Total Value Achieved', fontsize=12)
        ax.set_title('Total Value Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/16_optimization/05_resource_allocation/achievement_comparison.png',
                    dpi=300, bbox_inches='tight')
        print("Achievement visualization saved to: achievement_comparison.png")
        plt.show()


def main():
    """Main execution function."""
    print("="*60)
    print("RESOURCE ALLOCATION OPTIMIZATION")
    print("="*60)

    # Create optimizer
    optimizer = ResourceAllocationOptimizer(seed=42)

    # Generate problem
    problem = optimizer.generate_problem(n_projects=8, n_resources=4)

    # Solve using all methods
    results = optimizer.solve_all_methods(problem)

    # Compare results
    print("\n" + "="*60)
    print("Comparison of Methods")
    print("="*60)

    comparison_data = []
    for method_key, result in results.items():
        if result['success']:
            comparison_data.append({
                'Method': result['method'],
                'Total Value': f"{result['total_value']:.2f}",
                'Avg Achievement': f"{np.mean(result['achievement'])*100:.1f}%",
                'Min Achievement': f"{np.min(result['achievement'])*100:.1f}%",
                'Max Achievement': f"{np.max(result['achievement'])*100:.1f}%"
            })

    df_comparison = pd.DataFrame(comparison_data)
    print("\n", df_comparison.to_string(index=False))

    # Visualizations
    optimizer.visualize_allocation(problem)
    optimizer.visualize_achievement(problem)

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
