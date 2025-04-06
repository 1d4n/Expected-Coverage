# Expected Coverage over Hamming Graphs

This project investigates the expected number of steps required to cover the entire Hamming graph G = (V, E), where V = {0,1}^n and edges connect nodes with Hamming distance 1.

---

## Problem Description

Given the Hamming graph of order n, consider the following random process:

- At each step, a vertex v ∈ V is selected uniformly at random.
- The user receives the t-neighborhood of v, defined as:

  B_t(v) = { u ∈ V | d_H(u, v) ≤ t }

- The process continues until every vertex in the graph has been revealed, i.e., the union of all sampled neighborhoods covers V.

**Goal**:  
Estimate the expected number of steps required for full coverage of the graph, for varying values of:
- n – the dimension of the Hamming graph
- t – the Hamming radius

---

## Simulation & Analysis

The simulation performs multiple independent runs of the process and averages the number of steps until full coverage is achieved.

### Features:
- Coverage simulations for arbitrary n and t
- Support for large graphs using multiprocessing
- Visualization of:
  - Expected coverage time
  - Coverage rate as a function of time
- Curve fitting (linear, exponential, logarithmic)

---

## Running the Code

```bash
python main.py
```

Default experiments include:

- Coverage time as a function of n for fixed t
- Coverage time as a function of t for fixed n
- Coverage rate vs. steps for selected configurations

---

## Dependencies

Install with:

```bash
pip install numpy matplotlib scipy sympy
```

---
## Code Structure

The main logic is implemented in `main.py` and organized into the following components:

### Core Simulation Functions
- `simulate_experiment(n, t)`: Simulates the coverage process using explicit coverage tracking. Returns number of steps to full coverage.
- `simulate_experiment_large_t(n, t)`: Optimized for large t, tracks uncovered vertices for efficiency.
- `simulate_experiment_with_coverage_rate(n, t)`: Tracks coverage rate at each step (not just final result).

### Repeated Runs
- `repeat_experiment(n, t, runs)`: Runs the simulation multiple times and averages the number of steps.
- `repeat_experiment_coverage_rate(n, t, runs)`: Runs multiple simulations and collects coverage rate over time.

### Graph & Distance Tools
- `hamming_distance(u, v)`: Computes Hamming distance between two vertices.
- `ball_size(n, t)`: Calculates the number of nodes in the Hamming ball of radius t.
- `get_ball(n, t, v)`: Returns the set of nodes in the Hamming ball around vertex v.

### Coverage Helpers
- `cover_ball(covered, n, t, v)`: Marks nodes in the ball of v as covered.
- `get_new_covered(not_covered, t, v)`: Returns the set of newly covered nodes.

### Plotting & Visualization
- `generate_figure(...)`: Creates and saves coverage plots.
- `expectation_t_figure(n, runs)`: Plots expected steps as a function of t.
- `expectation_n_figure(n_values, runs, t)`: Plots expected steps as a function of n.
- `coverage_rate_steps_figure(n, runs, t)`: Plots coverage rate over time.

### t Parameter Models
- `ConstT(k)`: Fixed t value.
- `DivT(k)`: Sets t = n // k.
- `MinusT(k)`: Sets t = n - k.

### Curve Fitting Utilities
- `linear_regression(...)`
- `exponential_regression(...)`
- `logistic_regression(...)`

---

## Example Experiments

```python
expectation_n_figure(n_values=range(2, 14, 2), runs=1000, t=DivT(2))  # t = n / 2
expectation_n_figure(n_values=range(1, 13), runs=1000, t=ConstT(1))  # t = 1
expectation_t_figure(n=12, runs=1000)

coverage_rate_steps_figure(n=12, runs=1000, t=DivT(2))
coverage_rate_steps_figure(n=12, runs=1000, t=ConstT(1))
```