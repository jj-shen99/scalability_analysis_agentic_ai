# Scalability Visualization Module

## Overview
The Scalability Visualization Module provides a comprehensive set of plotting functions for visualizing system scalability characteristics. It creates publication-quality plots that help analyze and communicate performance patterns across different resource levels.

## Features
- **Basic Visualization Functions**:
  - Throughput vs. resource level plots
  - Response time vs. resource level plots
  - Speedup vs. resource level plots
  - Comparative throughput across configurations
  - Response time distribution analysis

- **Advanced Visualization Functions**:
  - Scalability efficiency analysis
  - Cost efficiency analysis
  - Efficiency heatmaps
  - Combined visualization dashboards

- **Plot Enhancements**:
  - Trend line fitting with equations
  - Reference lines for ideal scaling
  - SLA threshold indicators
  - Point annotations with specific values
  - Customizable styling and formatting

## Usage
This module is typically used as a component of the scalability analysis framework rather than directly. However, it can be used independently:

```python
from scalability_visualization_basic import (
    plot_throughput_vs_resource,
    plot_response_time_vs_resource,
    plot_speedup_vs_resource,
    create_basic_scalability_plots,
    create_advanced_scalability_plots
)

# Generate individual plots
throughput_plot = plot_throughput_vs_resource(
    resource_levels=[2, 4, 8, 16],
    throughputs=[200, 380, 720, 1350],
    output_dir='./plots'
)

# Generate all basic plots at once
plot_paths = create_basic_scalability_plots(
    resource_levels=[2, 4, 8, 16],
    throughputs=[200, 380, 720, 1350],
    response_times=[50, 55, 60, 65],
    output_dir='./plots'
)

# Generate advanced plots
advanced_plots = create_advanced_scalability_plots(
    resource_levels=[2, 4, 8, 16],
    throughputs=[200, 380, 720, 1350],
    output_dir='./plots',
    model_params={'amdahl': 0.95}  # Optional scalability model parameters
)
```

### Command Line Usage
```bash
python3 scalability_visualization_basic.py --files file1.jtl file2.jtl --levels 2 4 --output-dir ./plots [options]
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--files` | JTL files to analyze |
| `--levels` | Resource levels corresponding to files |
| `--output-dir` | Output directory for plots |
| `--advanced` | Generate advanced plots in addition to basic ones |
| `--show-plots` | Display plots instead of just saving them |

## Key Functions

### Basic Plotting Functions
- `plot_throughput_vs_resource()` - Generate throughput vs. resource level plot
- `plot_response_time_vs_resource()` - Generate response time vs. resource level plot
- `plot_speedup_vs_resource()` - Generate speedup vs. resource level plot
- `plot_comparative_throughput()` - Generate comparative throughput analysis
- `plot_response_time_distribution()` - Generate response time distribution comparison
- `create_basic_scalability_plots()` - Generate all basic plots at once

### Advanced Plotting Functions
- `plot_efficiency_heatmap()` - Generate scalability efficiency heatmap
- `plot_scalability_efficiency()` - Generate scalability efficiency analysis
- `plot_cost_efficiency()` - Generate cost efficiency analysis
- `create_advanced_scalability_plots()` - Generate all advanced plots at once

## Dependencies
- matplotlib - For creating plots
- seaborn - For enhanced plot styling
- numpy - For numerical calculations
- scalability_core - For JTL parsing when run directly
