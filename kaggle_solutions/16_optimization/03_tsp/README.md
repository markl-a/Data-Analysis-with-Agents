# Traveling Salesman Problem (TSP)

## Overview
This example demonstrates multiple approaches to solving the classic Traveling Salesman Problem (TSP), one of the most famous problems in combinatorial optimization and computer science.

## Problem Description

### The Traveling Salesman Problem
Given a set of cities and distances between them, find the shortest possible route that:
1. Visits each city exactly once
2. Returns to the starting city
3. Minimizes total travel distance

**Input:**
- N cities with known locations
- Distance matrix D where D[i,j] = distance from city i to city j

**Output:**
- A tour (permutation of cities) with minimum total distance

### Mathematical Formulation

```
Minimize: Σ Σ d[i,j] · x[i,j]

Subject to:
    Σ x[i,j] = 1    ∀j  (leave each city once)
    Σ x[i,j] = 1    ∀i  (enter each city once)
    Subtour elimination constraints
    x[i,j] ∈ {0, 1}     (binary decision)

Where:
    d[i,j] = distance from city i to city j
    x[i,j] = 1 if edge (i,j) is in tour, 0 otherwise
```

## Complexity

- **Problem Class**: NP-hard
- **Search Space**: (n-1)!/2 possible tours for n cities
- **Example**: 10 cities → 181,440 tours, 20 cities → 60,822,550,200,000,000 tours
- **Exact Solution**: Feasible only for small instances (n < 20)

## Methods Implemented

### 1. Brute Force (Exact)
Evaluates all possible tours to find the optimal solution.

**Algorithm:**
1. Generate all permutations of cities
2. Calculate length of each tour
3. Return shortest tour

**Complexity:** O(n!)
**Practical Limit:** ~10 cities
**Guarantee:** Finds optimal solution

### 2. Nearest Neighbor Heuristic
Greedy constructive heuristic.

**Algorithm:**
1. Start at a random city
2. Repeatedly visit the nearest unvisited city
3. Return to start when all visited

**Complexity:** O(n²)
**Quality:** Typically 15-25% above optimal
**Speed:** Very fast, suitable for large instances

### 3. 2-Opt Local Search
Improvement heuristic using edge swaps.

**Algorithm:**
1. Start with initial tour (e.g., nearest neighbor)
2. Try swapping pairs of edges
3. Accept swap if it improves tour
4. Repeat until no improvement possible

**Complexity:** O(n²) per iteration
**Quality:** Often within 5-10% of optimal
**Guarantee:** Local optimum (not global)

**How 2-Opt Works:**
```
Before:  A -- B    C -- D
After:   A -- C    B -- D
```
Reverses a segment of the tour.

### 4. Simulated Annealing
Metaheuristic based on thermodynamic annealing.

**Algorithm:**
1. Start with random tour
2. Generate neighbor by swapping cities
3. Accept if better, or with probability e^(-Δ/T)
4. Gradually decrease temperature T
5. Return best found

**Parameters:**
- Initial temperature: How likely to accept worse solutions initially
- Cooling rate: How fast temperature decreases
- Iterations: How long to search

**Complexity:** O(n · iterations)
**Quality:** Can find near-optimal solutions
**Advantage:** Escapes local optima

## Features

### Core Functionality
- Random problem generation
- Euclidean distance calculation
- Multiple solution algorithms
- Tour validation
- Performance comparison

### Visualizations
1. **Tour Maps**
   - City locations
   - Tour paths with arrows
   - Starting city marked
   - Separate plot for each method

2. **Performance Comparison**
   - Tour length bar chart
   - Computation time comparison
   - Best solution highlighted

3. **Convergence Analysis**
   - Simulated annealing convergence plot
   - Temperature decay visualization
   - Solution quality over iterations

## Key Concepts

### NP-Hardness
- No known polynomial-time exact algorithm
- Verification is easy, finding solution is hard
- Computational complexity grows exponentially

### Heuristic vs. Exact
- **Exact**: Guarantee optimal, slow for large instances
- **Heuristic**: Fast, good solutions, no optimality guarantee
- **Trade-off**: Quality vs. speed

### Local vs. Global Optimum
- **Local Optimum**: Best in neighborhood
- **Global Optimum**: Best overall
- Problem: Local search can get stuck

### Metaheuristics
Advanced techniques that:
- Escape local optima
- Balance exploration vs. exploitation
- Use randomization and memory

## Technical Implementation

### Dependencies
```python
numpy           # Numerical computations
pandas          # Data manipulation
matplotlib      # Visualization
scipy.spatial   # Distance calculations
```

### Key Classes
- `TSPSolver`: Main solver with multiple algorithms

### Algorithm Comparison

| Method | Time | Quality | Use Case |
|--------|------|---------|----------|
| Brute Force | O(n!) | Optimal | n ≤ 10 |
| Nearest Neighbor | O(n²) | ~80% optimal | Quick approximation |
| 2-Opt | O(n³) | ~90-95% optimal | Good balance |
| Simulated Annealing | O(n·iter) | ~95-99% optimal | Best quality heuristic |

## Results Interpretation

### Tour Quality
- Compare tour lengths across methods
- Calculate gap from optimal (if known)
- Identify method-specific patterns

### Computational Trade-offs
- Exact methods: Optimal but slow
- Fast heuristics: Quick but suboptimal
- Iterative methods: Balance of both

### When to Use Each Method
1. **Small instances (n < 15)**: Brute force or 2-opt
2. **Medium instances (15-100)**: 2-opt or simulated annealing
3. **Large instances (100+)**: Nearest neighbor or advanced methods
4. **Real-time applications**: Nearest neighbor
5. **Best quality needed**: Simulated annealing or genetic algorithms

## Usage

```bash
# Run with default settings (10 cities)
python solution.py
```

### Expected Output
1. Solutions from all applicable methods
2. Comparison table
3. Tour visualizations
4. Performance metrics
5. Convergence plots (for simulated annealing)

## Extensions

### Advanced Algorithms
1. **Genetic Algorithms**: Population-based evolution
2. **Ant Colony Optimization**: Pheromone-based search
3. **Branch and Bound**: Exact method with pruning
4. **Christofides Algorithm**: 1.5-approximation guarantee
5. **Lin-Kernighan Heuristic**: Advanced local search

### Problem Variants
1. **Multiple TSP**: Multiple salesmen
2. **TSP with Time Windows**: Deadline constraints
3. **Vehicle Routing Problem**: Capacity constraints
4. **Asymmetric TSP**: d[i,j] ≠ d[j,i]
5. **Clustered TSP**: City clusters

### Real-World Applications
- Logistics and delivery routing
- Circuit board drilling
- DNA sequencing
- Telescope positioning
- Manufacturing optimization
- Warehouse order picking

## Learning Objectives

After working through this example, you will understand:
1. What makes TSP NP-hard
2. Trade-offs between exact and heuristic methods
3. How local search works
4. Simulated annealing principles
5. When to use which algorithm
6. Performance vs. quality trade-offs

## Mathematical Background

### Hamiltonian Cycle
TSP is equivalent to finding minimum-weight Hamiltonian cycle:
- Hamiltonian Cycle: Visits each vertex exactly once
- Weight: Sum of edge distances
- TSP: Minimum-weight Hamiltonian cycle

### Approximation Algorithms
For metric TSP (triangle inequality holds):
- Nearest neighbor: No bound
- 2-approximation: MST-based algorithm
- 1.5-approximation: Christofides algorithm
- No PTAS unless P=NP

### Triangle Inequality
If d[i,j] ≤ d[i,k] + d[k,j] for all i,j,k:
- Called metric TSP
- Allows approximation algorithms
- Euclidean TSP always satisfies this

## Performance Notes

- **Brute force**: Only for demonstration (n ≤ 10)
- **Nearest neighbor**: Instant for hundreds of cities
- **2-opt**: Seconds for 50-100 cities
- **Simulated annealing**: Adjustable (more iterations = better quality)

## Common Pitfalls

1. **Symmetry**: TSP tours have rotational and reflectional symmetry
2. **Starting point**: Nearest neighbor quality depends on start
3. **Parameter tuning**: SA requires careful temperature scheduling
4. **Local optima**: 2-opt can get stuck
5. **Distance metric**: Different metrics give different solutions

## Benchmarking

### TSPLIB
Standard benchmark library:
- http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
- Known optimal solutions
- Various problem sizes
- Real-world instances

### Testing Your Implementation
Compare against:
- Optimal solutions for small instances
- Published heuristic results
- Other implementations

## Advanced Topics

### Held-Karp Algorithm
Dynamic programming exact algorithm:
- Complexity: O(n² · 2^n)
- Better than brute force
- Still exponential
- Practical for n ≤ 20

### Branch and Bound
Systematic search with pruning:
- Maintains lower/upper bounds
- Prunes hopeless branches
- Can solve instances with 50-100 cities

### Cutting Plane Methods
Integer programming approach:
- Start with LP relaxation
- Add violated inequalities
- Strengthen formulation
- State-of-the-art for exact solution

## Historical Notes

- First formulated in 1930s
- Dantzig, Fulkerson, Johnson solved 49-city instance (1954)
- Modern solvers can handle 1000+ cities
- Remains active research area

## References

- Applegate et al. (2006). The Traveling Salesman Problem: A Computational Study
- Lawler et al. (1985). The Traveling Salesman Problem
- TSPLIB: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/

This example provides a comprehensive introduction to TSP and fundamental optimization algorithm concepts applicable to many combinatorial problems.
