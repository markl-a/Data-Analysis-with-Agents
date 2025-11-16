# Integer Programming Optimization

## Overview
This example demonstrates Integer Programming (IP) and Mixed Integer Programming (MIP) techniques for solving discrete optimization problems. We implement a facility location problem with multiple solution approaches.

## Problem Description

### Facility Location Problem
A company needs to decide:
1. Which warehouses to open (binary decision)
2. How to assign customers to warehouses (binary decision)

**Decision Variables:**
- `y[i]`: Binary, 1 if warehouse i is opened, 0 otherwise
- `x[i,j]`: Binary, 1 if customer j is served by warehouse i, 0 otherwise

**Costs:**
- Fixed cost: Opening a warehouse
- Variable cost: Serving a customer from a warehouse

**Constraints:**
- Each customer must be served by exactly one warehouse
- Can only assign customers to open warehouses
- Warehouse capacity limits

### Mathematical Formulation

```
Minimize: Σ f[i]·y[i] + ΣΣ c[i,j]·x[i,j]

Subject to:
    Σ x[i,j] = 1                    ∀j (each customer served once)
    x[i,j] ≤ y[i]                   ∀i,j (assign only to open facilities)
    Σ d[j]·x[i,j] ≤ cap[i]·y[i]    ∀i (capacity constraints)
    y[i] ∈ {0, 1}                   ∀i (binary facility decisions)
    x[i,j] ∈ {0, 1}                 ∀i,j (binary assignment decisions)

Where:
    f[i] = fixed cost to open facility i
    c[i,j] = cost to serve customer j from facility i
    d[j] = demand of customer j
    cap[i] = capacity of facility i
```

## Methods Implemented

### 1. Branch and Bound (Exact Method)
- Uses SciPy's MILP solver
- Guarantees optimal solution
- Explores search tree systematically
- Pruning based on bounds

**How it works:**
1. Solve LP relaxation at root node
2. Branch on fractional variable
3. Solve subproblems recursively
4. Prune branches that can't improve best solution
5. Return optimal integer solution

### 2. Greedy Heuristic
- Fast approximation method
- Opens facilities in order of cost-effectiveness
- Assigns customers to cheapest available facility
- No optimality guarantee but quick

**Greedy Strategy:**
1. Calculate cost-effectiveness for each facility
2. Sort facilities by effectiveness
3. Open facilities and assign customers greedily
4. Continue until all customers served

### 3. LP Relaxation + Rounding
- Solve continuous relaxation
- Round fractional solutions to integers
- Re-optimize assignments
- Provides bounds on optimal solution

**Process:**
1. Solve LP relaxation (allow fractional values)
2. Round facility variables (y ≥ 0.5 → 1)
3. Reassign customers to open facilities
4. Calculate integrality gap

## Features

### Core Functionality
- Problem instance generation
- Multiple solution methods
- Constraint handling
- Feasibility verification
- Solution comparison

### Visualizations
1. **Facility Location Map**
   - Spatial layout of facilities and customers
   - Assignment connections
   - Open vs. closed facilities

2. **Cost Breakdown**
   - Pie chart of fixed vs. variable costs
   - Total cost summary

3. **Capacity Utilization**
   - Bar chart showing facility usage
   - Percentage of capacity used
   - Identification of underutilized facilities

4. **Assignment Cost Matrix**
   - Heatmap of assignment costs
   - Actual assignments marked
   - Visual identification of expensive routes

## Key Concepts

### Integer Programming
IP extends LP by requiring variables to take integer values:
- **Binary IP**: Variables in {0, 1}
- **Pure IP**: All variables are integers
- **Mixed IP (MIP)**: Some integer, some continuous

### Why IP is Hard
- NP-hard in general
- Cannot use Simplex directly
- Exponential worst-case complexity
- Optimal solution not always at vertex

### Branch and Bound
Core exact algorithm for IP:
- **Branching**: Partition problem into subproblems
- **Bounding**: Use LP relaxation for bounds
- **Pruning**: Eliminate suboptimal branches

### LP Relaxation
Continuous version of IP:
- Always provides lower bound (for minimization)
- Integrality gap: difference between IP and LP solutions
- Small gap → rounding likely works well

## Technical Implementation

### Dependencies
```python
numpy          # Numerical computations
pandas         # Data manipulation
matplotlib     # Visualization
scipy.optimize # MILP solver
```

### Key Classes
- `IntegerProgrammingSolver`: Main solver with multiple methods

### Algorithm Complexity
- **Branch and Bound**: Exponential worst-case O(2^n)
- **Greedy Heuristic**: O(n·m log m) where n=facilities, m=customers
- **LP Relaxation**: Polynomial + rounding overhead

## Results Interpretation

### Solution Quality Metrics
1. **Optimality Gap**: Difference from proven optimal
2. **Integrality Gap**: Difference from LP relaxation
3. **Computation Time**: Speed vs. quality trade-off

### Facility Selection
- Which facilities to open
- Why certain locations chosen
- Cost vs. coverage trade-offs

### Assignment Patterns
- Customer-facility matching
- Distance/cost minimization
- Load balancing across facilities

## Usage

```bash
# Run the complete analysis
python solution.py
```

### Expected Output
1. Solutions from all three methods
2. Comparison table
3. Facility location visualization
4. Cost breakdown analysis
5. Capacity utilization charts
6. Assignment cost matrices

## Extensions

### Possible Enhancements
1. **Multi-period planning**: Dynamic facility opening/closing
2. **Stochastic demand**: Uncertain customer requirements
3. **Multiple product types**: Different capacity requirements
4. **Network design**: Adding transportation network
5. **Budget constraints**: Limited capital for facility opening

### Advanced Techniques
- **Cutting planes**: Strengthen LP relaxation
- **Column generation**: For large-scale problems
- **Metaheuristics**: Genetic algorithms, simulated annealing
- **Lagrangian relaxation**: Decomposition approaches

### Real-World Applications
- Warehouse location
- Server placement
- ATM network design
- Emergency facility location
- Manufacturing plant siting
- Retail store location

## Learning Objectives

After working through this example, you will understand:
1. Difference between LP and IP
2. Why integrality makes problems harder
3. Branch and bound algorithm
4. Heuristic vs. exact methods
5. LP relaxation and rounding
6. Solution quality assessment
7. Trade-offs in facility location

## Mathematical Background

### Integrality Gap
For minimization problem:
```
Gap = (IP_optimal - LP_optimal) / LP_optimal × 100%
```
Small gap → LP relaxation is tight → rounding may work well

### Strong vs. Weak Formulations
- **Strong**: LP relaxation close to IP optimal
- **Weak**: Large integrality gap
- Reformulation can improve strength

### Facility Location Variants
1. **Uncapacitated**: No capacity limits (easier)
2. **Capacitated**: Capacity constraints (harder)
3. **p-median**: Open exactly p facilities
4. **p-center**: Minimize maximum distance

## Performance Notes

- **MILP Solver**: Best for small-medium instances (<100 variables)
- **Greedy**: Very fast, reasonable quality
- **Relaxation**: Good for bound computation
- **For large problems**: Use heuristics or specialized algorithms

## Common Pitfalls

1. **Large instances**: Exponential growth in solve time
2. **Weak formulation**: Poor LP bounds
3. **Symmetry**: Multiple equivalent solutions slow down search
4. **Numerical issues**: Floating-point precision with binary variables

## Advanced Topics

### Preprocessing
- Variable fixing
- Constraint tightening
- Symmetry breaking

### Valid Inequalities
- Cover cuts
- Knapsack cuts
- Gomory cuts

### Decomposition
- Benders decomposition
- Dantzig-Wolfe decomposition

## Validation

The example validates solutions by:
- Checking constraint satisfaction
- Verifying variable integrality
- Comparing across methods
- Visual inspection of assignments

## References

- Wolsey, L. A. (1998). Integer Programming
- Nemhauser, G. L., & Wolsey, L. A. (1988). Integer and Combinatorial Optimization
- SciPy Documentation: scipy.optimize.milp

## Benchmarking

Compare your solution quality:
- Small instances: Should find optimal within seconds
- Medium instances: May need minutes
- Large instances: Consider heuristics

This example provides foundation for tackling complex discrete optimization problems in operations research and logistics.
