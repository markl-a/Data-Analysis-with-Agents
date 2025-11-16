# Resource Allocation Optimization

## Overview
This example demonstrates multi-resource allocation optimization for project portfolio management, where limited resources must be distributed across competing projects to maximize overall value.

## Problem Description

### Multi-Resource Allocation Problem
An organization has multiple projects competing for limited resources (people, budget, equipment, time). The goal is to allocate resources to maximize total value while respecting constraints.

**Given:**
- N projects with expected values
- M resource types
- Resource requirements for each project
- Available quantities of each resource

**Goal:** Determine how much of each resource to allocate to each project

### Mathematical Formulation

```
Maximize: Σ v[i] · a[i]

Subject to:
    Σ r[i,j] · a[i] ≤ R[j]     ∀j (resource constraints)
    0 ≤ a[i] ≤ 1                ∀i (allocation fractions)

Where:
    v[i] = value/ROI of project i
    a[i] = allocation fraction for project i (0 = none, 1 = full)
    r[i,j] = units of resource j needed by project i
    R[j] = available units of resource j
```

## Methods Implemented

### 1. Proportional Allocation
Allocate resources proportionally to project value.

**Algorithm:**
- Calculate allocation fraction: f[i] = v[i] / Σv[i]
- Allocate resources accordingly
- Scale down if constraints violated

**Complexity:** O(n·m)
**Advantage:** Simple, fair in value terms
**Disadvantage:** May not be optimal

### 2. Linear Programming
Find optimal allocation using LP solver.

**Algorithm:**
- Formulate as LP problem
- Use simplex or interior point method
- Guarantee optimal solution

**Complexity:** Polynomial
**Advantage:** Provably optimal
**Disadvantage:** Doesn't consider fairness

### 3. Priority-Based Greedy
Allocate greedily based on project priorities.

**Algorithm:**
- Sort projects by priority
- Fully allocate to highest priority projects first
- Continue until resources exhausted

**Complexity:** O(n log n + n·m)
**Advantage:** Respects strategic priorities
**Disadvantage:** May waste resources, not optimal

### 4. Max-Min Fairness
Maximize the minimum allocation across projects.

**Algorithm:**
- Iteratively increase allocations
- Maintain equal allocation levels
- Continue until resource exhaustion

**Complexity:** O(n²·m)
**Advantage:** Fair, no project left behind
**Disadvantage:** May sacrifice total value

## Key Concepts

### Allocation vs. Selection
- **Selection**: Binary decision (fund or don't fund)
- **Allocation**: Continuous decision (how much to fund)
- This example focuses on allocation (0-100% funding)

### Fairness vs. Efficiency
- **Efficiency**: Maximize total value (LP approach)
- **Fairness**: Ensure equitable distribution (Max-Min)
- Trade-off between competing objectives

### Resource Constraints
- **Hard constraints**: Cannot be violated
- **Soft constraints**: Preferences, can be relaxed
- This example uses hard constraints

### Multi-Objective Optimization
Different stakeholders may have different goals:
- Executive: Maximize ROI
- Project managers: Maximize allocation
- Portfolio manager: Balance risk and return

## Features

### Core Functionality
- Random problem generation
- Multiple allocation strategies
- Constraint enforcement
- Performance comparison

### Visualizations
1. **Resource Allocation Charts**
   - Stacked bar charts per method
   - Resource breakdown by project
   - Visual comparison across methods

2. **Achievement Analysis**
   - Project funding levels
   - Method-by-method comparison
   - Achievement distribution

3. **Value Comparison**
   - Total value achieved per method
   - Highlight optimal solution
   - Trade-off visualization

## Technical Implementation

### Dependencies
```python
numpy          # Numerical computations
pandas         # Data manipulation
matplotlib     # Visualization
scipy.optimize # LP solver
```

### Algorithm Comparison

| Method | Objective | Time | Optimality | Fairness |
|--------|-----------|------|------------|----------|
| Proportional | Value-weighted | O(nm) | No | Medium |
| Linear Programming | Max total value | Poly | Yes | Low |
| Priority-Based | Strategic goals | O(n log n) | No | Low |
| Max-Min Fairness | Max minimum | O(n²m) | No | High |

## Usage

```bash
# Run with default settings (8 projects, 4 resources)
python solution.py
```

### Expected Output
1. Solutions from all four methods
2. Comparison table
3. Resource allocation visualizations
4. Achievement analysis charts
5. Performance metrics

## Real-World Applications

### Business Context
1. **R&D Portfolio**: Allocate research budget across projects
2. **IT Projects**: Distribute developers and infrastructure
3. **Marketing**: Allocate budget across campaigns
4. **Manufacturing**: Assign capacity to product lines
5. **Healthcare**: Distribute medical resources

### Organizational Levels
- **Strategic**: Long-term resource planning
- **Tactical**: Quarterly/annual budgeting
- **Operational**: Day-to-day resource scheduling

## Decision Criteria

### When to Use Each Method

**Proportional Allocation:**
- Quick first-pass allocation
- Value is clear metric
- No strong priorities

**Linear Programming:**
- Maximizing ROI is critical
- Resources are scarce
- Need provable optimality

**Priority-Based:**
- Strategic priorities dominate
- Political/organizational considerations
- Some projects must be funded

**Max-Min Fairness:**
- Equity is important
- Avoid abandoning projects
- Distribute opportunities

## Extensions

### Advanced Features
1. **Minimum Thresholds**: Projects need minimum funding to be viable
2. **Dependencies**: Some projects depend on others
3. **Time Windows**: Allocation over multiple periods
4. **Uncertainty**: Stochastic resource availability
5. **Multi-Objective**: Pareto-optimal solutions

### Additional Algorithms
1. **Genetic Algorithms**: For complex constraints
2. **Auction Mechanisms**: Market-based allocation
3. **Nash Bargaining**: Game-theoretic approach
4. **Robust Optimization**: Handle uncertainty

## Learning Objectives

After working through this example, you will understand:
1. Resource allocation problem formulation
2. Trade-offs between efficiency and fairness
3. LP application to resource problems
4. Greedy algorithms and limitations
5. Max-min fairness concept
6. Multi-objective optimization challenges

## Mathematical Background

### Pareto Efficiency
An allocation is Pareto efficient if no project can be improved without harming another.

LP solution is Pareto efficient for value objective.

### Nash Bargaining Solution
Game-theoretic approach maximizing product of utilities:
```
Maximize: Π (u[i] - u₀[i])
```
Balances efficiency and fairness.

### Shadow Prices
Dual variables from LP indicate value of additional resources:
- High shadow price → resource is bottleneck
- Zero shadow price → resource has slack

## Advanced Topics

### Uncertainty Modeling
- **Stochastic Programming**: Random parameters
- **Robust Optimization**: Worst-case scenarios
- **Chance Constraints**: Probabilistic satisfaction

### Dynamic Allocation
- Multi-period planning
- Resource carry-over
- Time-varying availability

### Market Mechanisms
- Auction-based allocation
- Price mechanisms
- Incentive compatibility

## Common Pitfalls

1. **Ignoring Complementarities**: Projects may have synergies
2. **Static Planning**: Reality changes, need re-allocation
3. **Fractional Funding**: Some projects need 100% or 0%
4. **Measurement Error**: Values and requirements are estimates
5. **Gaming**: People may inflate requirements

## Validation

### Solution Quality Checks
- All constraints satisfied
- Total allocation ≤ available
- Allocations in [0, 1]
- Compare against benchmarks

### Sensitivity Analysis
- How does solution change with:
  - Different resource levels
  - Changed project values
  - Varying priorities

## Performance Notes

- All methods scale to 100+ projects
- LP is bottleneck for very large instances
- Greedy methods are fastest
- Max-min fairness slower for many projects

## References

- Bertsimas, D., & Tsitsiklis, J. (1997). Introduction to Linear Optimization
- Kumar, A., & Kleinberg, J. (2000). Fairness measures for resource allocation
- Mas-Colell, A., et al. (1995). Microeconomic Theory

## Historical Context

Resource allocation has been studied since:
- 1940s: Linear programming developed
- 1970s: Fairness concepts formalized
- 1990s: Multi-objective optimization
- 2000s: Online algorithms for dynamic allocation

This example provides practical foundation for resource allocation in organizations, balancing competing objectives of efficiency, fairness, and strategic priorities.
