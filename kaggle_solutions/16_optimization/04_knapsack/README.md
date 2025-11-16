# Knapsack Problem Optimization

## Overview
This example demonstrates multiple approaches to solving the Knapsack Problem, one of the most fundamental problems in combinatorial optimization with numerous real-world applications.

## Problem Description

### The 0/1 Knapsack Problem
Given:
- A set of N items, each with weight w[i] and value v[i]
- A knapsack with weight capacity W

Goal: Select items to maximize total value without exceeding capacity

**Constraint:** Each item can be selected at most once (0/1 decision)

### Mathematical Formulation

```
Maximize: Σ v[i] · x[i]

Subject to:
    Σ w[i] · x[i] ≤ W  (weight constraint)
    x[i] ∈ {0, 1}       (binary decision)

Where:
    v[i] = value of item i
    w[i] = weight of item i
    W = knapsack capacity
    x[i] = 1 if item i is selected, 0 otherwise
```

## Complexity

- **Problem Class**: NP-hard
- **Exact Solution**: Pseudo-polynomial time via dynamic programming
- **Approximation**: FPTAS (Fully Polynomial-Time Approximation Scheme) exists

## Methods Implemented

### 1. Dynamic Programming (Optimal)
Classic DP solution guaranteeing optimal result.

**Algorithm:**
```
dp[i][w] = max value using first i items with capacity w

dp[i][w] = max(
    dp[i-1][w],                      // don't take item i
    dp[i-1][w-w[i]] + v[i]          // take item i
)
```

**Complexity:** O(n·W) time, O(n·W) space
**Guarantee:** Optimal solution
**Note:** Pseudo-polynomial (depends on W value)

### 2. Greedy by Value
Select items in decreasing order of value.

**Algorithm:**
1. Sort items by value (descending)
2. Add items while capacity allows
3. Stop when knapsack full or no items fit

**Complexity:** O(n log n)
**Quality:** No approximation guarantee
**Use:** Quick approximation

### 3. Greedy by Value/Weight Ratio
Select items by efficiency (value per unit weight).

**Algorithm:**
1. Calculate ratio r[i] = v[i] / w[i] for each item
2. Sort by ratio (descending)
3. Add items greedily

**Complexity:** O(n log n)
**Quality:** Better than greedy by value typically
**Note:** Optimal for fractional knapsack, not 0/1

### 4. Branch and Bound
Exact method using search tree with pruning.

**Algorithm:**
1. Explore binary tree (take/don't take)
2. Use fractional bound for pruning
3. Prune branches that can't improve best solution
4. Return optimal

**Complexity:** Exponential worst case, faster with good bounds
**Guarantee:** Optimal solution
**Advantage:** Can be faster than DP for small values/weights

## Features

### Core Functionality
- Random problem generation
- Multiple solution algorithms
- Solution validation
- Performance comparison
- Optimality verification

### Visualizations
1. **Item Selection Plot**
   - Scatter plot of weight vs. value
   - Selected items highlighted
   - Visual trade-offs

2. **Capacity Utilization**
   - Pie chart showing usage
   - Remaining capacity
   - Efficiency metric

3. **Item Characteristics**
   - Bar chart of weights and values
   - Selected items marked
   - Comparative analysis

4. **Efficiency Ratios**
   - Value/weight ratio for each item
   - Ranking visualization
   - Selection pattern analysis

5. **Method Comparison**
   - Total value comparison
   - Computation time analysis
   - Quality vs. speed trade-offs

## Key Concepts

### Greedy vs. Optimal
- Greedy heuristics are fast but not always optimal
- Example where greedy by value fails:
  - Items: (w=10, v=10), (w=9, v=9), (w=9, v=9)
  - Capacity: 18
  - Greedy by value: takes first item only (value=10)
  - Optimal: takes second and third (value=18)

### Dynamic Programming
- Breaks problem into subproblems
- Optimal substructure property
- Overlapping subproblems → memoization
- Bottom-up table filling

### Pseudo-Polynomial Time
- Complexity depends on numeric value (W), not just input size
- DP is O(nW), exponential in bits needed to represent W
- Still practical for reasonable capacities

### Approximation Schemes
For any ε > 0, FPTAS can find solution with value ≥ (1-ε)·OPT in time polynomial in n and 1/ε

## Technical Implementation

### Dependencies
```python
numpy       # Numerical computations
pandas      # Data manipulation
matplotlib  # Visualization
```

### Key Classes
- `KnapsackSolver`: Main solver with multiple algorithms

### Algorithm Comparison

| Method | Time | Space | Quality | Optimal |
|--------|------|-------|---------|---------|
| Dynamic Programming | O(nW) | O(nW) | 100% | Yes |
| Greedy Value | O(n log n) | O(1) | Variable | No |
| Greedy Ratio | O(n log n) | O(1) | Better | No |
| Branch & Bound | Exp* | O(n) | 100% | Yes |

*Exponential worst case, often faster than DP in practice

## Problem Variants

### 1. Fractional Knapsack
- Can take fractions of items
- Greedy by ratio is optimal
- Easier than 0/1 knapsack

### 2. Unbounded Knapsack
- Can take multiple copies of each item
- DP: dp[w] = max value with capacity w
- O(nW) time

### 3. Multiple Knapsacks
- Multiple bags with different capacities
- More complex, often use heuristics

### 4. Multidimensional Knapsack
- Multiple constraints (weight, volume, etc.)
- Much harder, no FPTAS

## Usage

```bash
# Run with default settings (20 items, capacity 100)
python solution.py
```

### Expected Output
1. Solutions from all methods
2. Comparison table
3. Item selection visualization
4. Capacity utilization analysis
5. Performance metrics

## Real-World Applications

### Direct Applications
- Resource allocation with budget constraints
- Project selection
- Cargo loading
- Investment portfolio selection

### Similar Problems
- Cutting stock problem
- Bin packing
- Subset sum
- Partition problem

### Industry Examples
1. **E-commerce**: Package optimization for shipping
2. **Finance**: Asset selection with capital constraint
3. **Manufacturing**: Production planning with resource limits
4. **Logistics**: Container loading optimization
5. **Cloud Computing**: VM placement with resource constraints

## Learning Objectives

After working through this example, you will understand:
1. Dynamic programming technique
2. Greedy algorithms and limitations
3. Branch and bound search
4. Pseudo-polynomial complexity
5. Trade-offs between exact and heuristic methods
6. When to use which approach

## Mathematical Background

### Optimal Substructure
If items {i₁, i₂, ..., iₖ} is optimal for capacity W:
- Either item n is included (optimal for W - w[n])
- Or item n is excluded (optimal for W with n-1 items)

### Recurrence Relation
```
K(i, w) = max{
    K(i-1, w),              if w[i] > w
    max(K(i-1, w), K(i-1, w-w[i]) + v[i])  otherwise
}
```

### Fractional Bound (for Branch & Bound)
```
Upper Bound = current_value + Σ (remaining_capacity / w[i]) · v[i]
```
Using fractional knapsack on remaining items.

## Performance Notes

- **DP**: Best for W ≤ 10,000
- **Branch & Bound**: Good for sparse problems
- **Greedy**: Use for quick approximations
- **For large W**: Consider approximation algorithms

## Common Pitfalls

1. **Integer Overflow**: Large values/weights
2. **Memory**: DP table can be huge
3. **Greedy Optimality**: Works for fractional, not 0/1
4. **Capacity Units**: Ensure integer or scale appropriately

## Advanced Topics

### Space Optimization
- 1D DP array: O(W) space
- Iterate backwards to avoid overwriting

### FPTAS Implementation
- Scale values to create pseudo-polynomial algorithm
- Trade accuracy for speed

### Core Extraction
- Identify items definitely in/out of solution
- Reduce problem size

## Extensions

### Code Enhancements
1. Implement unbounded knapsack
2. Add multiple knapsacks
3. Include item dependencies
4. Add value functions (non-linear)

### Advanced Algorithms
1. **Meet-in-the-middle**: O(2^(n/2)) exact
2. **Column generation**: For large instances
3. **Genetic algorithms**: For complex variants

## Validation Techniques

- Verify weight constraint satisfaction
- Compare DP and Branch & Bound (should match)
- Check greedy solutions are feasible
- Upper bound: fractional knapsack

## References

- Kellerer, H., Pferschy, U., & Pisinger, D. (2004). Knapsack Problems
- Martello, S., & Toth, P. (1990). Knapsack Problems: Algorithms and Computer Implementations
- Vazirani, V. V. (2001). Approximation Algorithms (Chapter on Knapsack)

## Historical Note

First studied in 1897. Named from problem of choosing what to carry in a knapsack. Despite extensive research, remains NP-hard with no known polynomial exact algorithm.

This example provides comprehensive coverage of knapsack problem-solving techniques applicable to resource allocation and optimization across many domains.
