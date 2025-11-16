# Linear Programming Optimization

## Overview
This example demonstrates comprehensive Linear Programming (LP) techniques for solving optimization problems. We implement and compare multiple solution methods for a production planning problem.

## Problem Description

### Production Planning Problem
A company manufactures two products (A and B) with limited resources:

**Products:**
- Product A: Profit $30 per unit
  - Requires 2 units of material
  - Requires 3 hours of labor

- Product B: Profit $40 per unit
  - Requires 3 units of material
  - Requires 2 hours of labor

**Constraints:**
- Material available: 120 units
- Labor available: 100 hours
- Non-negativity: Cannot produce negative quantities

**Objective:** Maximize total profit

### Mathematical Formulation

```
Maximize: Z = 30x₁ + 40x₂

Subject to:
    2x₁ + 3x₂ ≤ 120  (Material constraint)
    3x₁ + 2x₂ ≤ 100  (Labor constraint)
    x₁, x₂ ≥ 0       (Non-negativity)

Where:
    x₁ = units of Product A
    x₂ = units of Product B
```

## Methods Implemented

### 1. SciPy linprog (Simplex Method)
- Uses the HiGHS solver (modern implementation)
- Efficient and reliable for medium-sized problems
- Industry-standard optimization library

### 2. Graphical Method
- Visualizes the feasible region
- Finds corner points
- Evaluates objective function at each vertex
- Best for understanding 2D LP problems

### 3. Custom Simplex Algorithm
- Implementation from scratch
- Shows how the Simplex method works internally
- Educational value for understanding optimization

## Features

### Core Functionality
- Multiple solution approaches
- Constraint handling
- Optimality verification
- Iteration tracking

### Visualizations
1. **Feasible Region Plot**
   - Constraint lines
   - Feasible region shading
   - Optimal solution point
   - Isoprofit lines

2. **Resource Utilization**
   - Bar chart showing constraint usage
   - Percentage utilization
   - Identification of binding constraints

3. **Sensitivity Analysis**
   - Effect of changing objective coefficients
   - Impact of resource availability
   - Trade-off analysis

### Analysis Components
- Solution comparison across methods
- Sensitivity analysis
- Shadow prices (implicit)
- Resource utilization metrics

## Key Concepts

### Linear Programming
LP is a mathematical optimization technique where:
- Objective function is linear
- Constraints are linear inequalities/equalities
- Decision variables are continuous
- Optimal solution exists at corner points (vertices)

### Simplex Method
The Simplex algorithm:
1. Starts at a feasible corner point
2. Moves to adjacent corner points
3. Improves objective function value
4. Terminates at optimal solution

### Sensitivity Analysis
Examines how the optimal solution changes when:
- Objective coefficients vary
- Resource availability changes
- New constraints are added

## Technical Implementation

### Dependencies
```python
numpy          # Numerical computations
pandas         # Data manipulation
matplotlib     # Visualization
scipy.optimize # Linear programming solver
```

### Key Classes
- `LinearProgrammingSolver`: Main solver class with multiple methods

### Algorithm Complexity
- Simplex Method: O(m²n) average case
- Graphical Method: O(n²) for 2D problems
- Where m = constraints, n = variables

## Results Interpretation

### Optimal Solution
The solution will typically show:
- Optimal production quantities
- Maximum achievable profit
- Resource utilization percentages
- Binding vs. non-binding constraints

### Binding Constraints
A constraint is "binding" if:
- Used to full capacity at optimal solution
- Changing it would affect optimal value
- Shadow price (dual value) is positive

## Usage

```bash
# Run the complete analysis
python solution.py
```

### Expected Output
1. Solutions from all three methods
2. Comparison table
3. Feasible region visualization
4. Sensitivity analysis plots
5. Resource utilization charts

## Extensions

### Possible Enhancements
1. **Multi-product scenarios** (3+ products)
2. **Additional constraints** (storage, quality)
3. **Integer programming** (discrete quantities)
4. **Dual problem analysis**
5. **Parametric programming**

### Real-World Applications
- Manufacturing planning
- Resource allocation
- Diet optimization
- Transportation problems
- Blending problems
- Portfolio optimization (linear case)

## Learning Objectives

After working through this example, you will understand:
1. How to formulate LP problems mathematically
2. Multiple methods for solving LP problems
3. Interpretation of optimal solutions
4. Sensitivity analysis techniques
5. Resource utilization and bottlenecks
6. When LP is appropriate vs. other optimization methods

## Mathematical Background

### Fundamental Theorem of LP
- If an LP has an optimal solution, at least one optimal solution occurs at a corner point
- This justifies the Simplex method's vertex-to-vertex movement

### Duality
Every LP has a dual problem:
- Primal: maximize profit subject to resource constraints
- Dual: minimize resource costs subject to profit requirements
- Optimal values are equal (strong duality)

## Performance Notes

- **SciPy method**: Fastest for production use
- **Graphical method**: Limited to 2D problems, educational value
- **Custom Simplex**: Slower but shows algorithm mechanics

## References

- Dantzig, G. B. (1963). Linear Programming and Extensions
- Vanderbei, R. J. (2014). Linear Programming: Foundations and Extensions
- SciPy Documentation: scipy.optimize.linprog

## Common Pitfalls

1. **Unbounded problems**: No optimal solution (profit → ∞)
2. **Infeasible problems**: No solution satisfies all constraints
3. **Degeneracy**: Multiple optimal solutions
4. **Numerical precision**: Floating-point errors in large problems

## Validation

The example includes validation by:
- Comparing results across methods
- Verifying constraint satisfaction
- Checking optimality conditions
- Visual inspection of feasible region
