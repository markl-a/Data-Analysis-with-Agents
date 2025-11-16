# Scheduling Optimization

## Overview
This example demonstrates job shop scheduling optimization, a fundamental problem in operations research with applications in manufacturing, project management, and resource planning.

## Problem Description

### Job Shop Scheduling Problem
Schedule a set of jobs on multiple machines to minimize total completion time (makespan) while respecting:
- Operation precedence within jobs
- Machine capacity (one job at a time)
- Processing time requirements

**Given:**
- N jobs, each with sequence of operations
- M machines
- Processing time for each operation
- Machine assignment for each operation

**Goal:** Assign start times to minimize makespan

### Mathematical Formulation

```
Minimize: Cmax (makespan)

Subject to:
    s[i,k+1] ≥ s[i,k] + p[i,k]              ∀i,k (precedence)
    s[i,k] ≥ s[j,k] + p[j,k]  OR
    s[j,k] ≥ s[i,k] + p[i,k]                ∀i≠j on same machine (no overlap)
    s[i,k] ≥ 0                               ∀i,k (non-negative)
    Cmax ≥ s[i,last] + p[i,last]            ∀i (makespan definition)

Where:
    s[i,k] = start time of operation k of job i
    p[i,k] = processing time of operation k of job i
    Cmax = makespan (completion time of last job)
```

## Complexity

- **Problem Class**: NP-hard (even for 2 machines)
- **Optimal Solution**: Difficult for n > 10
- **Practical Approach**: Use heuristics for large instances

## Methods Implemented

### 1. First-Come-First-Served (FCFS)
Process jobs in the order they arrive.

**Algorithm:**
- Jobs processed in order 1, 2, ..., n
- Each operation scheduled ASAP
- Simple, no optimization

**Complexity:** O(n·m)
**Quality:** Baseline, often poor
**Use:** Quick initial schedule

### 2. Shortest Processing Time (SPT)
Process jobs with shortest total time first.

**Algorithm:**
- Calculate total processing time per job
- Sort jobs by total time (ascending)
- Schedule in that order

**Complexity:** O(n log n + n·m)
**Quality:** Good for minimizing average completion
**Advantage:** Reduces work-in-progress

### 3. Priority-Based
Schedule high-priority jobs first.

**Algorithm:**
- Jobs have assigned priorities
- Sort by priority (descending)
- Schedule in priority order

**Complexity:** O(n log n + n·m)
**Quality:** Depends on priorities
**Use:** When strategic importance varies

### 4. Earliest Due Date (EDD)
Minimize tardiness by scheduling urgent jobs first.

**Algorithm:**
- Sort jobs by due date (ascending)
- Schedule in that order
- Minimizes maximum tardiness

**Complexity:** O(n log n + n·m)
**Quality:** Optimal for single machine, good heuristic for multiple
**Objective:** Minimize lateness

## Key Concepts

### Makespan
Total time from start of first job to completion of last job:
- Minimize idle time
- Maximize throughput
- Key metric for capacity planning

### Machine Utilization
Percentage of time machines are busy:
```
Utilization = (Total Processing Time) / (Makespan × Machines)
```
High utilization → efficient schedule

### Tardiness
Lateness beyond due date:
```
Tardiness[i] = max(0, Completion[i] - DueDate[i])
```
Zero if on-time, positive if late

### Bottleneck Machine
Machine with highest utilization:
- Limits overall throughput
- Focus optimization efforts here
- Consider additional capacity

## Features

### Core Functionality
- Random problem generation
- Multiple scheduling heuristics
- Gantt chart visualization
- Performance metrics
- Utilization analysis

### Visualizations
1. **Gantt Charts**
   - Timeline view per method
   - Color-coded by job
   - Machine assignment
   - Visual schedule comparison

2. **Makespan Comparison**
   - Bar chart of completion times
   - Best method highlighted
   - Performance differences

3. **Utilization Analysis**
   - Machine usage percentage
   - Idle time identification
   - Efficiency comparison

## Technical Implementation

### Dependencies
```python
numpy       # Numerical computations
pandas      # Data manipulation
matplotlib  # Visualization (including Gantt charts)
```

### Algorithm Comparison

| Method | Objective | Time | Quality | Use Case |
|--------|-----------|------|---------|----------|
| FCFS | None | O(nm) | Poor | Baseline |
| SPT | Avg completion | O(n log n) | Good | WIP reduction |
| Priority | Strategic goals | O(n log n) | Variable | Business priorities |
| EDD | Tardiness | O(n log n) | Good | Meet deadlines |

## Usage

```bash
# Run with default settings (10 jobs, 3 machines)
python solution.py
```

### Expected Output
1. Solutions from all four methods
2. Comparison table
3. Gantt chart for best method
4. Performance comparison charts
5. Utilization analysis

## Real-World Applications

### Manufacturing
- Production scheduling
- Assembly line planning
- Batch processing
- Maintenance scheduling

### Project Management
- Task scheduling
- Resource allocation
- Critical path management
- Milestone planning

### Services
- Operating room scheduling
- Appointment scheduling
- Classroom assignment
- Server task scheduling

### Logistics
- Truck loading/unloading
- Warehouse operations
- Delivery route optimization
- Airport gate assignment

## Decision Criteria

### Choosing a Heuristic

**Use FCFS when:**
- Fairness is critical
- Processing order is predetermined
- Need simple implementation

**Use SPT when:**
- Minimizing average completion time
- Reducing work-in-progress
- Small jobs should finish quickly

**Use Priority when:**
- Jobs have different importance
- Strategic considerations dominate
- VIP customers/projects

**Use EDD when:**
- Meeting deadlines is critical
- Tardiness penalties exist
- Customer commitments

## Extensions

### Advanced Algorithms
1. **Critical Path Method (CPM)**: Project scheduling
2. **Johnson's Algorithm**: Optimal for 2-machine flow shop
3. **Tabu Search**: Metaheuristic for large instances
4. **Genetic Algorithms**: Population-based optimization
5. **Branch and Bound**: Exact method for small problems

### Problem Variants
1. **Flow Shop**: Jobs visit machines in same order
2. **Open Shop**: No precedence constraints
3. **Flexible Job Shop**: Operations can use alternative machines
4. **Parallel Machines**: Multiple identical machines
5. **Setup Times**: Time to switch between jobs

### Additional Constraints
- Setup times between jobs
- Machine breakdowns
- Limited buffers
- Release dates
- Preemption allowed/forbidden

## Learning Objectives

After working through this example, you will understand:
1. Job shop scheduling problem structure
2. Common scheduling heuristics
3. Trade-offs between different objectives
4. Gantt chart interpretation
5. Machine utilization concepts
6. When to use which scheduling rule

## Mathematical Background

### Disjunctive Graph
Job shop can be represented as:
- Nodes = operations
- Conjunctive arcs = precedence within jobs
- Disjunctive arcs = machine conflicts
- Finding schedule = selecting disjunctive arc directions

### Critical Path
Longest path in precedence graph:
- Determines makespan
- Identifies bottlenecks
- No slack in critical operations

### Lower Bounds
Useful for evaluating heuristics:
- Machine bound: max(total time on any machine)
- Job bound: max(total time for any job)
- Optimal ≥ max(machine bound, job bound)

## Performance Notes

- All heuristics scale to 100+ jobs easily
- FCFS is fastest
- Priority and EDD require sorting
- Gantt chart generation is O(n·m)

## Common Pitfalls

1. **Ignoring Setup Times**: Real systems have changeover costs
2. **Static Schedules**: Reality is dynamic, need rescheduling
3. **Single Objective**: Often multiple objectives matter
4. **Deterministic Assumptions**: Processing times vary
5. **No Disruptions**: Machines break, jobs arrive late

## Advanced Topics

### Dynamic Scheduling
- Real-time rescheduling
- Online algorithms
- Rolling horizon
- Event-driven updates

### Robust Scheduling
- Handle uncertainty
- Buffer times
- Scenario analysis
- Stochastic processing times

### Multi-Objective Optimization
Balance competing goals:
- Minimize makespan
- Minimize tardiness
- Maximize utilization
- Minimize WIP

## Validation

### Schedule Quality Checks
- All precedence constraints satisfied
- No machine conflicts
- All operations scheduled
- Compare against lower bounds

### Metrics
- Makespan (primary)
- Average completion time
- Tardiness
- Utilization
- Idle time

## References

- Pinedo, M. (2016). Scheduling: Theory, Algorithms, and Systems
- Brucker, P. (2007). Scheduling Algorithms
- Baker, K. R., & Trietsch, D. (2013). Principles of Sequencing and Scheduling

## Historical Context

- 1950s: First scheduling algorithms (Johnson)
- 1960s: Complexity results (NP-hardness proven)
- 1970s: Branch and bound methods
- 1980s-90s: Metaheuristics (SA, GA, TS)
- 2000s+: Online and stochastic scheduling

## Benchmarking

Test against standard instances:
- Taillard's benchmark problems
- Compare makespan to best known
- Evaluate computation time
- Test scalability

This example provides practical foundation for scheduling problems across manufacturing, services, and project management domains.
