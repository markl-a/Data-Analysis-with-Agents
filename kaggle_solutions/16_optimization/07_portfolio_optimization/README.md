# Portfolio Optimization

## Overview
This example demonstrates portfolio optimization using Modern Portfolio Theory (MPT), developed by Harry Markowitz in 1952. We explore the fundamental trade-off between risk and return in investment portfolios.

## Problem Description

### Portfolio Optimization Problem
Allocate capital across multiple assets to achieve investment objectives:
- Maximize return for given risk level
- Minimize risk for given return level
- Maximize risk-adjusted return (Sharpe ratio)

**Given:**
- N assets with expected returns
- Covariance matrix of returns (risk)
- Investment constraints

**Goal:** Find optimal portfolio weights

### Mathematical Formulation

**Mean-Variance Optimization:**
```
Minimize: w^T Σ w  (portfolio variance)

Subject to:
    w^T μ ≥ r_target     (target return constraint)
    Σ w_i = 1             (fully invested)
    w_i ≥ 0               (no short selling)

Where:
    w = vector of portfolio weights
    μ = vector of expected returns
    Σ = covariance matrix
    r_target = target return
```

**Maximum Sharpe Ratio:**
```
Maximize: (w^T μ - r_f) / √(w^T Σ w)

Subject to:
    Σ w_i = 1
    w_i ≥ 0

Where:
    r_f = risk-free rate
```

## Methods Implemented

### 1. Equal Weight (1/n)
Naive diversification: invest equally in all assets.

**Formula:** w_i = 1/n for all i

**Advantages:**
- Simple, no estimation needed
- Surprisingly competitive
- No optimization error

**Disadvantages:**
- Ignores expected returns
- Ignores correlations
- May be suboptimal

### 2. Minimum Variance Portfolio
Lowest-risk portfolio on efficient frontier.

**Optimization:**
- Minimize: w^T Σ w
- Subject to: Σw_i = 1, w_i ≥ 0

**Characteristics:**
- Lowest possible risk
- May have low return
- Very conservative

**Use When:**
- Risk minimization is primary goal
- Risk-averse investors
- Uncertain return estimates

### 3. Maximum Sharpe Ratio Portfolio
Best risk-adjusted return (tangency portfolio).

**Sharpe Ratio:** SR = (R_p - R_f) / σ_p

**Characteristics:**
- Optimal for mean-variance investors
- Tangent to efficient frontier
- Balances risk and return

**Use When:**
- Risk-adjusted return matters
- Rational investor framework
- Capital allocation line

### 4. Target Return Portfolio
Minimum risk for specified return level.

**Optimization:**
- Minimize: w^T Σ w
- Subject to: w^T μ = r_target, Σw_i = 1, w_i ≥ 0

**Characteristics:**
- Guarantees minimum return
- Minimizes risk for that return
- Point on efficient frontier

**Use When:**
- Return requirement is known
- Liability matching
- Goal-based investing

## Key Concepts

### Modern Portfolio Theory
MPT foundation:
1. Investors are risk-averse
2. Investors maximize utility
3. Returns are normally distributed
4. Only mean and variance matter

### Efficient Frontier
Set of portfolios that:
- Maximize return for given risk
- Minimize risk for given return
- No dominated portfolios

All efficient portfolios lie on this frontier.

### Diversification
"Don't put all eggs in one basket"
- Reduces unsystematic risk
- Correlation matters
- Benefits diminish with more assets

**Diversification Benefit:**
```
σ_portfolio < Weighted average of σ_individual
```

### Sharpe Ratio
Risk-adjusted return metric:
```
SR = (R_p - R_f) / σ_p
```

- Higher is better
- Compares portfolios fairly
- Most widely used metric

### Risk Measures
1. **Volatility (σ)**: Standard deviation of returns
2. **Variance (σ²)**: Squared volatility
3. **Covariance**: Joint variability
4. **Correlation**: Normalized covariance

## Features

### Core Functionality
- Synthetic market data generation
- Multiple optimization strategies
- Efficient frontier computation
- Risk-return analysis
- Performance metrics

### Visualizations
1. **Portfolio Allocations**
   - Bar chart comparing weights
   - Asset distribution across strategies
   - Concentration analysis

2. **Risk-Return Scatter**
   - Efficient frontier curve
   - Portfolio positions
   - Individual assets
   - Optimal portfolios marked

3. **Sharpe Ratio Comparison**
   - Performance ranking
   - Best portfolio highlighted
   - Risk-adjusted returns

4. **Allocation Pie Chart**
   - Best portfolio breakdown
   - Visual weight distribution
   - Concentration visualization

## Technical Implementation

### Dependencies
```python
numpy          # Matrix operations
pandas         # Data manipulation
matplotlib     # Visualization
scipy.optimize # Optimization solvers
```

### Optimization Methods
- **SLSQP**: Sequential Least Squares Programming
- **Quadratic Programming**: For variance minimization
- **Constrained Optimization**: Equality and inequality constraints

### Algorithm Comparison

| Method | Objective | Risk | Return | Sharpe | Use Case |
|--------|-----------|------|--------|--------|----------|
| Equal Weight | None | Medium | Medium | Medium | Benchmark |
| Min Variance | Min risk | Lowest | Low | Low | Conservative |
| Max Sharpe | Max SR | Medium | High | Highest | Rational |
| Target Return | Min risk @ target | Varies | Fixed | Varies | Goal-based |

## Usage

```bash
# Run with default settings (8 assets)
python solution.py
```

### Expected Output
1. Asset characteristics
2. Solutions from all four methods
3. Comparison table
4. Risk-return visualization
5. Efficient frontier plot
6. Allocation analysis

## Real-World Applications

### Investment Management
- Mutual fund construction
- ETF design
- Pension fund allocation
- Endowment management

### Personal Finance
- Retirement planning
- Savings allocation
- 401(k) optimization
- Robo-advisors

### Corporate Finance
- Treasury management
- Cash allocation
- Risk management
- Capital budgeting

### Institutional
- Insurance portfolios
- Sovereign wealth funds
- University endowments
- Foundation investments

## Extensions

### Advanced Features
1. **Transaction Costs**: Include trading fees
2. **Constraints**: Sector limits, position limits
3. **Rebalancing**: Dynamic portfolio adjustment
4. **Multi-Period**: Time-varying optimization
5. **Robust Optimization**: Parameter uncertainty

### Alternative Models
1. **Black-Litterman**: Incorporate views
2. **Risk Parity**: Equal risk contribution
3. **Minimum CVaR**: Conditional Value at Risk
4. **Maximum Diversification**: Decorrelation
5. **Hierarchical Risk Parity**: Clustering-based

### Risk Models
1. **Factor Models**: Fama-French, etc.
2. **GARCH**: Time-varying volatility
3. **Copulas**: Non-normal dependencies
4. **Downside Risk**: Semi-variance, VaR

## Learning Objectives

After working through this example, you will understand:
1. Modern Portfolio Theory foundations
2. Risk-return trade-off
3. Diversification benefits
4. Efficient frontier concept
5. Sharpe ratio optimization
6. Portfolio optimization techniques
7. When to use which strategy

## Mathematical Background

### Markowitz Model
Original MPT formulation (1952):
```
Minimize: σ_p² = w^T Σ w
Subject to: w^T μ = r_target, w^T 1 = 1
```

### Capital Market Line (CML)
Combination of risk-free asset and market portfolio:
```
E[R_p] = r_f + σ_p · (E[R_m] - r_f) / σ_m
```

### Two-Fund Separation
Any optimal portfolio is a combination of:
1. Risk-free asset
2. Tangency portfolio (max Sharpe)

### Lagrangian Optimization
Portfolio optimization via Lagrange multipliers:
```
L = w^T Σ w - λ₁(w^T μ - r) - λ₂(w^T 1 - 1)
```

## Common Pitfalls

1. **Estimation Error**: Returns are uncertain
2. **Optimization Error**: Small errors amplified
3. **Parameter Instability**: Estimates change over time
4. **Extreme Weights**: Unconstrained optimization
5. **Normal Distribution**: Returns aren't normal
6. **Static Model**: Markets evolve
7. **Ignoring Costs**: Transaction costs matter

## Advanced Topics

### Robust Portfolio Optimization
Handle parameter uncertainty:
- Worst-case optimization
- Bayesian approaches
- Resampling methods
- Shrinkage estimators

### Multi-Objective Optimization
Multiple goals beyond mean-variance:
- Maximize return
- Minimize risk
- Minimize tracking error
- Maximize Sharpe ratio
- Pareto-optimal solutions

### Dynamic Portfolio Management
- Rebalancing strategies
- Time-varying constraints
- Stochastic control
- Multi-period optimization

## Performance Metrics

### Ex-Ante (Expected)
- Expected return
- Expected volatility
- Expected Sharpe ratio

### Ex-Post (Realized)
- Actual return
- Realized volatility
- Information ratio
- Maximum drawdown
- Sortino ratio

## Validation

### Backtesting
- Historical simulation
- Out-of-sample testing
- Walk-forward analysis
- Monte Carlo simulation

### Sensitivity Analysis
- Parameter variations
- Constraint changes
- Different time periods
- Robustness checks

## References

### Foundational Papers
- Markowitz, H. (1952). "Portfolio Selection". Journal of Finance
- Sharpe, W. F. (1964). "Capital Asset Pricing Model"
- Tobin, J. (1958). "Liquidity Preference as Behavior Towards Risk"

### Books
- Markowitz, H. M. (1959). Portfolio Selection: Efficient Diversification
- Elton, E. J., et al. (2014). Modern Portfolio Theory and Investment Analysis

### Modern Extensions
- Black, F., & Litterman, R. (1992). "Global Portfolio Optimization"
- Maillard, S., et al. (2010). "On the Properties of Equally-Weighted Risk Contributions Portfolios"

## Historical Context

- **1952**: Markowitz introduces MPT
- **1964**: Sharpe develops CAPM
- **1976**: Ross proposes APT
- **1990s**: Practical implementation grows
- **2000s**: Robust and Bayesian methods
- **2010s**: Machine learning integration

## Benchmarking

Compare against:
- Market index (S&P 500)
- Equal-weight portfolio
- Risk parity
- Target date funds
- Professionally managed funds

## Practical Considerations

### Implementation
- Rebalancing frequency
- Transaction costs
- Tax efficiency
- Liquidity constraints
- Regulatory requirements

### Monitoring
- Performance attribution
- Risk decomposition
- Drift from target
- Constraint violations

This example provides a comprehensive foundation for portfolio optimization, applicable to personal investing, institutional asset management, and financial advisory services.
