"""
Integer Programming Solutions
==============================

This example demonstrates Integer Programming (IP) and Mixed Integer Programming (MIP)
techniques for discrete optimization problems.

Problem: Facility Location Problem
A company needs to decide which warehouses to open and how to assign customers to them.
- Fixed cost to open each warehouse
- Variable cost to serve each customer from each warehouse
- Each customer must be served by exactly one warehouse
- Goal: Minimize total cost

Author: Kaggle Solutions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import milp, LinearConstraint, Bounds
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class IntegerProgrammingSolver:
    """
    Comprehensive Integer Programming solver with multiple approaches.
    """

    def __init__(self):
        """Initialize the IP solver."""
        self.results = {}
        np.random.seed(42)

    def generate_facility_problem(self, n_facilities=5, n_customers=10) -> Dict:
        """
        Generate a facility location problem instance.

        Args:
            n_facilities: Number of potential facility locations
            n_customers: Number of customers to serve

        Returns:
            Dictionary with problem data
        """
        # Fixed costs to open facilities
        fixed_costs = np.random.randint(800, 1500, n_facilities)

        # Assignment costs (cost to serve customer j from facility i)
        assignment_costs = np.random.randint(50, 200, (n_facilities, n_customers))

        # Facility capacities
        capacities = np.random.randint(3, 6, n_facilities)

        # Customer demands
        demands = np.ones(n_customers)

        problem = {
            'n_facilities': n_facilities,
            'n_customers': n_customers,
            'fixed_costs': fixed_costs,
            'assignment_costs': assignment_costs,
            'capacities': capacities,
            'demands': demands,
            'facility_names': [f'Warehouse {i+1}' for i in range(n_facilities)],
            'customer_names': [f'Customer {i+1}' for i in range(n_customers)]
        }

        return problem

    def solve_branch_and_bound(self, problem: Dict) -> Dict:
        """
        Solve using Branch and Bound algorithm (via scipy MILP).

        Args:
            problem: Problem definition dictionary

        Returns:
            Dictionary with solution results
        """
        print("\n" + "="*60)
        print("Method 1: Branch and Bound (SciPy MILP)")
        print("="*60)

        nf = problem['n_facilities']
        nc = problem['n_customers']

        # Decision variables:
        # y[i] = 1 if facility i is open, 0 otherwise (0 to nf-1)
        # x[i,j] = 1 if customer j is served by facility i, 0 otherwise (nf to nf+nf*nc-1)

        # Objective coefficients
        c = np.concatenate([
            problem['fixed_costs'],  # Facility opening costs
            problem['assignment_costs'].flatten()  # Assignment costs
        ])

        # Integer constraints: all variables are binary
        integrality = np.ones(len(c), dtype=int)

        # Bounds: all variables are binary [0, 1]
        bounds = Bounds(lb=np.zeros(len(c)), ub=np.ones(len(c)))

        # Constraints
        constraints = []

        # 1. Each customer must be served by exactly one facility
        for j in range(nc):
            A_row = np.zeros(len(c))
            for i in range(nf):
                A_row[nf + i * nc + j] = 1
            constraints.append(LinearConstraint(A_row, lb=1, ub=1))

        # 2. Can only assign customer to open facility: x[i,j] <= y[i]
        for i in range(nf):
            for j in range(nc):
                A_row = np.zeros(len(c))
                A_row[i] = -1  # -y[i]
                A_row[nf + i * nc + j] = 1  # x[i,j]
                constraints.append(LinearConstraint(A_row, lb=-np.inf, ub=0))

        # 3. Capacity constraints
        for i in range(nf):
            A_row = np.zeros(len(c))
            for j in range(nc):
                A_row[nf + i * nc + j] = problem['demands'][j]
            A_row[i] = -problem['capacities'][i]  # -capacity * y[i]
            constraints.append(LinearConstraint(A_row, lb=-np.inf, ub=0))

        # Solve
        from scipy.optimize import milp
        result = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)

        # Extract solution
        y = result.x[:nf]
        x = result.x[nf:].reshape(nf, nc)

        solution = {
            'method': 'Branch and Bound',
            'y': y,
            'x': x,
            'optimal_value': result.fun if result.success else None,
            'success': result.success,
            'message': result.message,
            'open_facilities': np.where(y > 0.5)[0]
        }

        if result.success:
            print(f"Status: {result.message}")
            print(f"\nFacilities to open: {[problem['facility_names'][i] for i in solution['open_facilities']]}")
            print(f"Total cost: ${result.fun:.2f}")
            print(f"\nAssignments:")
            for j in range(nc):
                for i in range(nf):
                    if x[i, j] > 0.5:
                        print(f"  {problem['customer_names'][j]} -> {problem['facility_names'][i]} "
                              f"(cost: ${problem['assignment_costs'][i, j]})")
        else:
            print(f"Failed: {result.message}")

        return solution

    def solve_greedy_heuristic(self, problem: Dict) -> Dict:
        """
        Solve using a greedy heuristic approach.

        Args:
            problem: Problem definition dictionary

        Returns:
            Dictionary with solution results
        """
        print("\n" + "="*60)
        print("Method 2: Greedy Heuristic")
        print("="*60)

        nf = problem['n_facilities']
        nc = problem['n_customers']

        # Calculate cost-effectiveness of each facility
        # (average assignment cost / fixed cost ratio)
        avg_assignment_cost = problem['assignment_costs'].mean(axis=1)
        effectiveness = avg_assignment_cost / (problem['fixed_costs'] + 1)

        # Sort facilities by effectiveness (ascending - lower is better)
        facility_order = np.argsort(effectiveness)

        y = np.zeros(nf)
        x = np.zeros((nf, nc))
        assigned = np.zeros(nc, dtype=bool)
        remaining_capacity = problem['capacities'].copy()

        print("\nGreedy selection process:")

        # Try to assign customers to facilities in order
        for facility_idx in facility_order:
            if all(assigned):
                break

            # Open this facility if it can serve unassigned customers
            customers_to_assign = []
            temp_capacity = remaining_capacity[facility_idx]

            # Sort customers by cost for this facility
            unassigned = np.where(~assigned)[0]
            costs = [(j, problem['assignment_costs'][facility_idx, j]) for j in unassigned]
            costs.sort(key=lambda x: x[1])

            for j, cost in costs:
                if temp_capacity >= problem['demands'][j]:
                    customers_to_assign.append(j)
                    temp_capacity -= problem['demands'][j]

            if customers_to_assign:
                y[facility_idx] = 1
                for j in customers_to_assign:
                    x[facility_idx, j] = 1
                    assigned[j] = True
                    remaining_capacity[facility_idx] -= problem['demands'][j]

                print(f"  Opened {problem['facility_names'][facility_idx]}: "
                      f"Serving {len(customers_to_assign)} customers")

        # Calculate total cost
        total_cost = (
            np.sum(y * problem['fixed_costs']) +
            np.sum(x * problem['assignment_costs'])
        )

        solution = {
            'method': 'Greedy Heuristic',
            'y': y,
            'x': x,
            'optimal_value': total_cost,
            'success': all(assigned),
            'message': 'All customers assigned' if all(assigned) else 'Some customers not assigned',
            'open_facilities': np.where(y > 0.5)[0]
        }

        print(f"\nTotal cost: ${total_cost:.2f}")
        print(f"Facilities opened: {len(solution['open_facilities'])}")

        return solution

    def solve_relaxation_rounding(self, problem: Dict) -> Dict:
        """
        Solve LP relaxation and round the solution.

        Args:
            problem: Problem definition dictionary

        Returns:
            Dictionary with solution results
        """
        print("\n" + "="*60)
        print("Method 3: LP Relaxation + Rounding")
        print("="*60)

        from scipy.optimize import linprog

        nf = problem['n_facilities']
        nc = problem['n_customers']

        # Same formulation as Branch and Bound, but allow fractional values
        c = np.concatenate([
            problem['fixed_costs'],
            problem['assignment_costs'].flatten()
        ])

        # Constraints
        A_eq = []
        b_eq = []
        A_ub = []
        b_ub = []

        # Each customer must be served exactly once
        for j in range(nc):
            A_row = np.zeros(len(c))
            for i in range(nf):
                A_row[nf + i * nc + j] = 1
            A_eq.append(A_row)
            b_eq.append(1)

        # Assignment only if facility is open
        for i in range(nf):
            for j in range(nc):
                A_row = np.zeros(len(c))
                A_row[i] = -1
                A_row[nf + i * nc + j] = 1
                A_ub.append(A_row)
                b_ub.append(0)

        # Capacity constraints
        for i in range(nf):
            A_row = np.zeros(len(c))
            for j in range(nc):
                A_row[nf + i * nc + j] = problem['demands'][j]
            A_row[i] = -problem['capacities'][i]
            A_ub.append(A_row)
            b_ub.append(0)

        # Solve LP relaxation
        bounds = [(0, 1) for _ in range(len(c))]
        result = linprog(c=c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                        bounds=bounds, method='highs')

        if result.success:
            print(f"LP Relaxation optimal value: ${result.fun:.2f}")

            # Extract and round solution
            y_relaxed = result.x[:nf]
            x_relaxed = result.x[nf:].reshape(nf, nc)

            # Rounding: open facilities with y >= 0.5
            y = (y_relaxed >= 0.5).astype(float)

            # Reassign customers to open facilities (greedy)
            x = np.zeros((nf, nc))
            open_facilities = np.where(y > 0.5)[0]

            for j in range(nc):
                # Assign to cheapest open facility with capacity
                best_facility = None
                best_cost = np.inf

                for i in open_facilities:
                    if x[i, :].sum() < problem['capacities'][i]:
                        if problem['assignment_costs'][i, j] < best_cost:
                            best_cost = problem['assignment_costs'][i, j]
                            best_facility = i

                if best_facility is not None:
                    x[best_facility, j] = 1

            total_cost = (
                np.sum(y * problem['fixed_costs']) +
                np.sum(x * problem['assignment_costs'])
            )

            solution = {
                'method': 'LP Relaxation + Rounding',
                'y': y,
                'x': x,
                'optimal_value': total_cost,
                'relaxed_value': result.fun,
                'success': True,
                'message': 'Solution rounded successfully',
                'open_facilities': np.where(y > 0.5)[0]
            }

            print(f"Rounded solution cost: ${total_cost:.2f}")
            print(f"Integrality gap: ${total_cost - result.fun:.2f}")

        else:
            solution = {
                'method': 'LP Relaxation + Rounding',
                'success': False,
                'message': result.message
            }

        return solution

    def solve_all_methods(self, problem: Dict):
        """Solve the problem using all available methods."""
        print("\nSOLVING INTEGER PROGRAMMING PROBLEM")
        print("="*60)
        print("Facility Location Problem:")
        print(f"  Facilities: {problem['n_facilities']}")
        print(f"  Customers: {problem['n_customers']}")
        print(f"  Fixed costs: ${problem['fixed_costs']}")
        print(f"  Capacities: {problem['capacities']}")
        print("="*60)

        # Solve with different methods
        self.results['milp'] = self.solve_branch_and_bound(problem)
        self.results['greedy'] = self.solve_greedy_heuristic(problem)
        self.results['relaxation'] = self.solve_relaxation_rounding(problem)

        return self.results

    def visualize_solution(self, problem: Dict, solution: Dict):
        """
        Visualize the facility location solution.

        Args:
            problem: Problem definition dictionary
            solution: Solution dictionary
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Facility locations and assignments
        ax = axes[0, 0]

        # Generate random locations for visualization
        np.random.seed(42)
        facility_locs = np.random.rand(problem['n_facilities'], 2) * 100
        customer_locs = np.random.rand(problem['n_customers'], 2) * 100

        # Plot facilities
        open_facilities = solution['open_facilities']
        closed_facilities = np.setdiff1d(range(problem['n_facilities']), open_facilities)

        ax.scatter(facility_locs[open_facilities, 0], facility_locs[open_facilities, 1],
                  s=500, c='red', marker='s', label='Open Facilities',
                  edgecolors='black', linewidths=2, zorder=3)
        ax.scatter(facility_locs[closed_facilities, 0], facility_locs[closed_facilities, 1],
                  s=300, c='gray', marker='s', label='Closed Facilities',
                  alpha=0.5, edgecolors='black', linewidths=1, zorder=2)

        # Plot customers
        ax.scatter(customer_locs[:, 0], customer_locs[:, 1],
                  s=200, c='blue', marker='o', label='Customers',
                  edgecolors='black', linewidths=1, zorder=2)

        # Plot assignments
        colors = plt.cm.tab10(np.linspace(0, 1, len(open_facilities)))
        for idx, i in enumerate(open_facilities):
            for j in range(problem['n_customers']):
                if solution['x'][i, j] > 0.5:
                    ax.plot([facility_locs[i, 0], customer_locs[j, 0]],
                           [facility_locs[i, 1], customer_locs[j, 1]],
                           'k-', alpha=0.3, linewidth=1, zorder=1)

        # Label facilities and customers
        for i in range(problem['n_facilities']):
            ax.annotate(f'W{i+1}', xy=facility_locs[i], fontsize=9,
                       ha='center', va='center', fontweight='bold', color='white')

        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_title('Facility Locations and Customer Assignments', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Plot 2: Cost breakdown
        ax = axes[0, 1]

        fixed_cost_total = np.sum(solution['y'] * problem['fixed_costs'])
        assignment_cost_total = np.sum(solution['x'] * problem['assignment_costs'])

        costs = [fixed_cost_total, assignment_cost_total]
        labels = ['Fixed Costs', 'Assignment Costs']
        colors_pie = ['#ff6b6b', '#4ecdc4']

        wedges, texts, autotexts = ax.pie(costs, labels=labels, autopct='%1.1f%%',
                                          colors=colors_pie, startangle=90,
                                          textprops={'fontsize': 12, 'fontweight': 'bold'})

        ax.set_title(f'Cost Breakdown\nTotal: ${solution["optimal_value"]:.2f}',
                    fontsize=14, fontweight='bold')

        # Plot 3: Capacity utilization
        ax = axes[1, 0]

        utilization = []
        facility_labels = []
        for i in solution['open_facilities']:
            used = solution['x'][i, :].sum()
            capacity = problem['capacities'][i]
            utilization.append((used / capacity) * 100)
            facility_labels.append(problem['facility_names'][i])

        bars = ax.barh(facility_labels, utilization, color='green', alpha=0.7)
        ax.axvline(x=100, color='r', linestyle='--', linewidth=2, label='100% Capacity')

        for i, (bar, util) in enumerate(zip(bars, utilization)):
            ax.text(util + 2, i, f'{util:.1f}%', va='center', fontsize=10, fontweight='bold')

        ax.set_xlim(0, 110)
        ax.set_xlabel('Utilization (%)', fontsize=12)
        ax.set_title('Facility Capacity Utilization', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')

        # Plot 4: Assignment cost matrix
        ax = axes[1, 1]

        # Show only open facilities
        cost_matrix = problem['assignment_costs'][solution['open_facilities'], :]
        assignment_matrix = solution['x'][solution['open_facilities'], :]

        im = ax.imshow(cost_matrix, cmap='YlOrRd', aspect='auto')

        # Mark assignments with X
        for i in range(len(solution['open_facilities'])):
            for j in range(problem['n_customers']):
                if assignment_matrix[i, j] > 0.5:
                    ax.text(j, i, 'X', ha='center', va='center',
                           fontsize=16, fontweight='bold', color='blue')

        ax.set_xticks(range(problem['n_customers']))
        ax.set_yticks(range(len(solution['open_facilities'])))
        ax.set_xticklabels([f'C{i+1}' for i in range(problem['n_customers'])])
        ax.set_yticklabels([problem['facility_names'][i] for i in solution['open_facilities']])
        ax.set_xlabel('Customers', fontsize=12)
        ax.set_ylabel('Open Facilities', fontsize=12)
        ax.set_title('Assignment Costs and Decisions', fontsize=14, fontweight='bold')

        plt.colorbar(im, ax=ax, label='Cost ($)')

        plt.tight_layout()
        plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/16_optimization/02_integer_programming/ip_visualization.png',
                    dpi=300, bbox_inches='tight')
        print("\nVisualization saved to: ip_visualization.png")
        plt.show()


def main():
    """Main execution function."""
    print("="*60)
    print("INTEGER PROGRAMMING OPTIMIZATION")
    print("="*60)

    # Create solver
    solver = IntegerProgrammingSolver()

    # Generate problem
    problem = solver.generate_facility_problem(n_facilities=5, n_customers=10)

    # Solve using all methods
    results = solver.solve_all_methods(problem)

    # Compare results
    print("\n" + "="*60)
    print("Comparison of Methods")
    print("="*60)

    comparison_data = []
    for method_name, result in results.items():
        if result['success']:
            comparison_data.append({
                'Method': result['method'],
                'Facilities Opened': len(result['open_facilities']),
                'Total Cost': f"${result['optimal_value']:.2f}",
                'Status': 'Success'
            })
        else:
            comparison_data.append({
                'Method': result['method'],
                'Facilities Opened': '-',
                'Total Cost': '-',
                'Status': 'Failed'
            })

    df_comparison = pd.DataFrame(comparison_data)
    print("\n", df_comparison.to_string(index=False))

    # Visualize best solution
    best_solution = results['milp'] if results['milp']['success'] else results['greedy']
    solver.visualize_solution(problem, best_solution)

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
