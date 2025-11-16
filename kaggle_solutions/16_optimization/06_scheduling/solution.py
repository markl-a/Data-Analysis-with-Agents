"""
Scheduling Optimization
=======================

This example demonstrates various scheduling optimization techniques for
job shop scheduling, a classic operations research problem.

Problem: Schedule jobs on machines to minimize total completion time (makespan)
while respecting precedence constraints and resource availability.

Author: Kaggle Solutions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Dict, List, Tuple
import time
import warnings
warnings.filterwarnings('ignore')


class SchedulingOptimizer:
    """
    Job shop scheduling optimizer with multiple algorithms.
    """

    def __init__(self, seed=42):
        """Initialize the scheduler."""
        self.seed = seed
        np.random.seed(seed)
        self.results = {}

    def generate_problem(self, n_jobs=10, n_machines=3) -> Dict:
        """
        Generate a job shop scheduling problem.

        Args:
            n_jobs: Number of jobs
            n_machines: Number of machines

        Returns:
            Problem dictionary
        """
        # Processing times for each job on each machine
        processing_times = np.random.randint(5, 20, (n_jobs, n_machines))

        # Machine order for each job (some jobs may skip machines)
        machine_order = []
        for i in range(n_jobs):
            order = list(range(n_machines))
            np.random.shuffle(order)
            machine_order.append(order)

        # Job priorities (higher = more important)
        priorities = np.random.randint(1, 6, n_jobs)

        # Due dates
        total_work = np.sum(processing_times, axis=1)
        due_dates = total_work * np.random.uniform(1.5, 2.5, n_jobs)

        problem = {
            'n_jobs': n_jobs,
            'n_machines': n_machines,
            'processing_times': processing_times,
            'machine_order': machine_order,
            'priorities': priorities,
            'due_dates': due_dates,
            'job_names': [f'Job {i+1}' for i in range(n_jobs)],
            'machine_names': [f'Machine {chr(65+i)}' for i in range(n_machines)]
        }

        return problem

    def calculate_makespan(self, schedule: Dict, problem: Dict) -> float:
        """
        Calculate the makespan (total completion time) of a schedule.

        Args:
            schedule: Schedule dictionary
            problem: Problem dictionary

        Returns:
            Makespan value
        """
        completion_times = []
        for job_id in range(problem['n_jobs']):
            if job_id in schedule:
                job_schedule = schedule[job_id]
                if job_schedule:
                    max_end = max(task['end'] for task in job_schedule)
                    completion_times.append(max_end)

        return max(completion_times) if completion_times else 0

    def solve_fcfs(self, problem: Dict) -> Dict:
        """
        Solve using First-Come-First-Served (FCFS) strategy.

        Args:
            problem: Problem dictionary

        Returns:
            Solution dictionary
        """
        print("\n" + "="*60)
        print("Method 1: First-Come-First-Served (FCFS)")
        print("="*60)

        start_time = time.time()

        n_jobs = problem['n_jobs']
        n_machines = problem['n_machines']

        # Track when each machine becomes available
        machine_available = np.zeros(n_machines)

        # Track when each job's previous operation completes
        job_available = np.zeros(n_jobs)

        # Schedule: dict[job_id] = list of operations
        schedule = {i: [] for i in range(n_jobs)}

        # Process jobs in order
        for job_id in range(n_jobs):
            for machine_idx in problem['machine_order'][job_id]:
                proc_time = problem['processing_times'][job_id][machine_idx]

                # Start time is max of machine and job availability
                start = max(machine_available[machine_idx], job_available[job_id])
                end = start + proc_time

                schedule[job_id].append({
                    'machine': machine_idx,
                    'start': start,
                    'end': end,
                    'duration': proc_time
                })

                machine_available[machine_idx] = end
                job_available[job_id] = end

        makespan = self.calculate_makespan(schedule, problem)
        elapsed_time = time.time() - start_time

        solution = {
            'method': 'FCFS',
            'schedule': schedule,
            'makespan': makespan,
            'success': True,
            'time': elapsed_time
        }

        print(f"Makespan: {makespan:.2f}")
        print(f"Time: {elapsed_time:.4f} seconds")

        return solution

    def solve_spt(self, problem: Dict) -> Dict:
        """
        Solve using Shortest Processing Time (SPT) first.

        Args:
            problem: Problem dictionary

        Returns:
            Solution dictionary
        """
        print("\n" + "="*60)
        print("Method 2: Shortest Processing Time (SPT)")
        print("="*60)

        start_time = time.time()

        n_jobs = problem['n_jobs']
        n_machines = problem['n_machines']

        # Sort jobs by total processing time
        total_times = np.sum(problem['processing_times'], axis=1)
        job_order = np.argsort(total_times)

        machine_available = np.zeros(n_machines)
        job_available = np.zeros(n_jobs)
        schedule = {i: [] for i in range(n_jobs)}

        print("\nJob order (by total processing time):")
        print([problem['job_names'][i] for i in job_order])

        for job_id in job_order:
            for machine_idx in problem['machine_order'][job_id]:
                proc_time = problem['processing_times'][job_id][machine_idx]

                start = max(machine_available[machine_idx], job_available[job_id])
                end = start + proc_time

                schedule[job_id].append({
                    'machine': machine_idx,
                    'start': start,
                    'end': end,
                    'duration': proc_time
                })

                machine_available[machine_idx] = end
                job_available[job_id] = end

        makespan = self.calculate_makespan(schedule, problem)
        elapsed_time = time.time() - start_time

        solution = {
            'method': 'SPT',
            'schedule': schedule,
            'makespan': makespan,
            'success': True,
            'time': elapsed_time
        }

        print(f"Makespan: {makespan:.2f}")
        print(f"Time: {elapsed_time:.4f} seconds")

        return solution

    def solve_priority(self, problem: Dict) -> Dict:
        """
        Solve using priority-based scheduling.

        Args:
            problem: Problem dictionary

        Returns:
            Solution dictionary
        """
        print("\n" + "="*60)
        print("Method 3: Priority-Based Scheduling")
        print("="*60)

        start_time = time.time()

        n_jobs = problem['n_jobs']
        n_machines = problem['n_machines']

        # Sort jobs by priority (descending)
        job_order = np.argsort(-problem['priorities'])

        machine_available = np.zeros(n_machines)
        job_available = np.zeros(n_jobs)
        schedule = {i: [] for i in range(n_jobs)}

        print("\nJob order (by priority):")
        for job_id in job_order:
            print(f"  {problem['job_names'][job_id]}: Priority {problem['priorities'][job_id]}")

        for job_id in job_order:
            for machine_idx in problem['machine_order'][job_id]:
                proc_time = problem['processing_times'][job_id][machine_idx]

                start = max(machine_available[machine_idx], job_available[job_id])
                end = start + proc_time

                schedule[job_id].append({
                    'machine': machine_idx,
                    'start': start,
                    'end': end,
                    'duration': proc_time
                })

                machine_available[machine_idx] = end
                job_available[job_id] = end

        makespan = self.calculate_makespan(schedule, problem)
        elapsed_time = time.time() - start_time

        solution = {
            'method': 'Priority',
            'schedule': schedule,
            'makespan': makespan,
            'success': True,
            'time': elapsed_time
        }

        print(f"Makespan: {makespan:.2f}")
        print(f"Time: {elapsed_time:.4f} seconds")

        return solution

    def solve_edd(self, problem: Dict) -> Dict:
        """
        Solve using Earliest Due Date (EDD) first.

        Args:
            problem: Problem dictionary

        Returns:
            Solution dictionary
        """
        print("\n" + "="*60)
        print("Method 4: Earliest Due Date (EDD)")
        print("="*60)

        start_time = time.time()

        n_jobs = problem['n_jobs']
        n_machines = problem['n_machines']

        # Sort jobs by due date
        job_order = np.argsort(problem['due_dates'])

        machine_available = np.zeros(n_machines)
        job_available = np.zeros(n_jobs)
        schedule = {i: [] for i in range(n_jobs)}

        print("\nJob order (by due date):")
        for job_id in job_order:
            print(f"  {problem['job_names'][job_id]}: Due {problem['due_dates'][job_id]:.1f}")

        for job_id in job_order:
            for machine_idx in problem['machine_order'][job_id]:
                proc_time = problem['processing_times'][job_id][machine_idx]

                start = max(machine_available[machine_idx], job_available[job_id])
                end = start + proc_time

                schedule[job_id].append({
                    'machine': machine_idx,
                    'start': start,
                    'end': end,
                    'duration': proc_time
                })

                machine_available[machine_idx] = end
                job_available[job_id] = end

        makespan = self.calculate_makespan(schedule, problem)
        elapsed_time = time.time() - start_time

        # Calculate tardiness
        tardiness = []
        for job_id in range(n_jobs):
            completion = max(task['end'] for task in schedule[job_id])
            tardy = max(0, completion - problem['due_dates'][job_id])
            tardiness.append(tardy)

        total_tardiness = sum(tardiness)

        solution = {
            'method': 'EDD',
            'schedule': schedule,
            'makespan': makespan,
            'tardiness': total_tardiness,
            'success': True,
            'time': elapsed_time
        }

        print(f"Makespan: {makespan:.2f}")
        print(f"Total Tardiness: {total_tardiness:.2f}")
        print(f"Time: {elapsed_time:.4f} seconds")

        return solution

    def solve_all_methods(self, problem: Dict):
        """Solve using all methods."""
        print("\nSCHEDULING OPTIMIZATION")
        print("="*60)
        print(f"Jobs: {problem['n_jobs']}")
        print(f"Machines: {problem['n_machines']}")
        print("="*60)

        self.results['fcfs'] = self.solve_fcfs(problem)
        self.results['spt'] = self.solve_spt(problem)
        self.results['priority'] = self.solve_priority(problem)
        self.results['edd'] = self.solve_edd(problem)

        return self.results

    def visualize_gantt_chart(self, problem: Dict, solution: Dict):
        """
        Create Gantt chart visualization of schedule.

        Args:
            problem: Problem dictionary
            solution: Solution dictionary
        """
        fig, ax = plt.subplots(figsize=(16, 8))

        schedule = solution['schedule']
        colors = plt.cm.Set3(np.linspace(0, 1, problem['n_jobs']))

        # Plot each job's operations
        for job_id, job_schedule in schedule.items():
            for task in job_schedule:
                machine = task['machine']
                start = task['start']
                duration = task['duration']

                # Draw rectangle for task
                rect = Rectangle((start, machine - 0.4), duration, 0.8,
                               facecolor=colors[job_id], edgecolor='black',
                               linewidth=2)
                ax.add_patch(rect)

                # Add job label
                ax.text(start + duration/2, machine, f'J{job_id+1}',
                       ha='center', va='center', fontsize=10,
                       fontweight='bold', color='black')

        # Configure axes
        ax.set_xlim(0, solution['makespan'] * 1.05)
        ax.set_ylim(-0.5, problem['n_machines'] - 0.5)
        ax.set_yticks(range(problem['n_machines']))
        ax.set_yticklabels(problem['machine_names'])
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Machines', fontsize=12)
        ax.set_title(f'{solution["method"]} Schedule - Makespan: {solution["makespan"]:.2f}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # Add legend
        legend_elements = [Rectangle((0, 0), 1, 1, facecolor=colors[i],
                                    edgecolor='black', label=problem['job_names'][i])
                          for i in range(problem['n_jobs'])]
        ax.legend(handles=legend_elements, loc='upper right',
                 bbox_to_anchor=(1.15, 1), ncol=1)

        plt.tight_layout()
        filename = f'/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/16_optimization/06_scheduling/gantt_{solution["method"].lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Gantt chart saved to: gantt_{solution['method'].lower().replace(' ', '_')}.png")
        plt.show()

    def visualize_comparison(self, problem: Dict):
        """Compare all scheduling methods."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        methods = [v['method'] for v in self.results.values()]
        makespans = [v['makespan'] for v in self.results.values()]
        times = [v['time'] for v in self.results.values()]

        # Plot 1: Makespan comparison
        ax = axes[0]
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

        bars = ax.bar(methods, makespans, color=colors, alpha=0.7,
                     edgecolor='black', linewidth=2)

        # Highlight best
        best_idx = np.argmin(makespans)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(4)

        for bar, ms in zip(bars, makespans):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{ms:.1f}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_ylabel('Makespan (time units)', fontsize=12)
        ax.set_title('Makespan Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 2: Machine utilization
        ax = axes[1]

        utilizations = []
        for method_key, result in self.results.items():
            schedule = result['schedule']
            makespan = result['makespan']

            # Calculate total busy time per machine
            machine_busy = np.zeros(problem['n_machines'])
            for job_id, job_schedule in schedule.items():
                for task in job_schedule:
                    machine_busy[task['machine']] += task['duration']

            # Average utilization
            avg_util = np.mean(machine_busy / makespan) * 100
            utilizations.append(avg_util)

        bars = ax.bar(methods, utilizations, color=colors, alpha=0.7,
                     edgecolor='black', linewidth=2)

        for bar, util in zip(bars, utilizations):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{util:.1f}%',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_ylabel('Average Machine Utilization (%)', fontsize=12)
        ax.set_title('Machine Utilization Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=100, color='r', linestyle='--', alpha=0.5)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/16_optimization/06_scheduling/scheduling_comparison.png',
                    dpi=300, bbox_inches='tight')
        print("\nComparison visualization saved to: scheduling_comparison.png")
        plt.show()


def main():
    """Main execution function."""
    print("="*60)
    print("SCHEDULING OPTIMIZATION")
    print("="*60)

    # Create optimizer
    optimizer = SchedulingOptimizer(seed=42)

    # Generate problem
    problem = optimizer.generate_problem(n_jobs=10, n_machines=3)

    # Solve using all methods
    results = optimizer.solve_all_methods(problem)

    # Compare results
    print("\n" + "="*60)
    print("Comparison of Methods")
    print("="*60)

    comparison_data = []
    for method_key, result in results.items():
        comparison_data.append({
            'Method': result['method'],
            'Makespan': f"{result['makespan']:.2f}",
            'Time (s)': f"{result['time']:.4f}"
        })

    df_comparison = pd.DataFrame(comparison_data)
    print("\n", df_comparison.to_string(index=False))

    # Find best
    best_method = min(results.items(), key=lambda x: x[1]['makespan'])
    print(f"\nBest method: {best_method[1]['method']} with makespan {best_method[1]['makespan']:.2f}")

    # Visualize best solution
    optimizer.visualize_gantt_chart(problem, best_method[1])

    # Compare all methods
    optimizer.visualize_comparison(problem)

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
