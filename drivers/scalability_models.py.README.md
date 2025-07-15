# Scalability Models Module

## Overview
The Scalability Models module implements various theoretical models for analyzing system scalability. It provides functions for fitting observed performance data to these models, generating predictions, creating visualizations, and interpreting results.

## Features
- Implementation of key scalability models:
  - Amdahl's Law - Theoretical speedup with fixed-size problems
  - Gustafson's Law - Theoretical speedup with scaled problems
  - Universal Scalability Law (USL) - Advanced model accounting for contention and coherency costs
- Functions for fitting observed data to these models
- Visualization tools for comparing actual vs. theoretical scaling
- Functions for generating theoretical projections beyond measured data
- Interpretation utilities for explaining results in human-readable terms
- Optimization suggestion generation based on model parameters

## Usage
This module is typically used as a library imported by higher-level analysis scripts rather than directly. However, it can be used independently for scalability modeling:

```python
from scalability_models import (
    amdahls_law, fit_amdahls_law,
    gustafsons_law, fit_gustafsons_law,
    universal_scalability_law, fit_universal_scalability_law,
    plot_scalability_models, interpret_scalability_results
)

# Example with observed data
resource_levels = [1, 2, 4, 8]
speedups = [1.0, 1.9, 3.5, 6.0]

# Fit data to models
p_amdahl, error_amdahl = fit_amdahls_law(resource_levels, speedups)
sigma, kappa, error_usl = fit_universal_scalability_law(resource_levels, speedups)

# Generate interpretation
models = {'amdahl': p_amdahl, 'usl': (sigma, kappa)}
interpretations = interpret_scalability_results(models)

# Create visualization
plot_paths = plot_scalability_models(
    resource_levels, speedups, 
    output_dir='./plots', 
    models=models
)
```

## Key Functions

### Model Implementations
- `amdahls_law(p, n)` - Calculate speedup using Amdahl's Law
- `gustafsons_law(p, n)` - Calculate speedup using Gustafson's Law
- `universal_scalability_law(sigma, kappa, n)` - Calculate speedup using USL

### Model Fitting
- `fit_amdahls_law(resource_levels, speedups)` - Find optimal parameters for Amdahl's Law
- `fit_gustafsons_law(resource_levels, speedups)` - Find optimal parameters for Gustafson's Law
- `fit_universal_scalability_law(resource_levels, speedups)` - Find optimal parameters for USL

### Visualization
- `plot_scalability_models(resource_levels, actual_speedups, output_dir, models, show_plots)` - Generate comparison plots
- `plot_theoretical_projections(resource_levels, actual_speedups, output_dir, theoretical_projections, max_projection, show_plots)` - Generate plots with projections

### Interpretation
- `interpret_scalability_results(models)` - Generate human-readable interpretations
- `suggest_optimizations(models, interpretations)` - Generate optimization suggestions

## Model Parameters

### Amdahl's & Gustafson's Laws
- `p`: Fraction of the program that is parallelizable (between 0 and 1)

### Universal Scalability Law
- `sigma`: Contention/serialization factor (between 0 and 1)
- `kappa`: Coherency delay factor (between 0 and 1)

## Theoretical Background
- **Amdahl's Law**: S(n) = 1 / ((1 - p) + p/n)
- **Gustafson's Law**: S(n) = (1 - p) + p × n
- **Universal Scalability Law**: S(n) = n / (1 + σ(n-1) + κn(n-1))

Where:
- S(n) = speedup with n resources
- p = parallelizable fraction
- σ (sigma) = contention factor
- κ (kappa) = coherency delay factor
