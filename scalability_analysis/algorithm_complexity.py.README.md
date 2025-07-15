# Algorithm Complexity Analysis Module

## Overview
This module provides functions to analyze and identify the algorithmic complexity of a system based on performance data across different load levels. It fits various complexity models (O(1), O(log n), O(n), etc.) to performance data and identifies the most likely algorithmic complexity pattern.

## Features
- Analyzes performance data to determine algorithmic complexity
- Supports multiple complexity models:
  - O(1) - Constant time
  - O(log n) - Logarithmic time
  - O(n) - Linear time
  - O(n log n) - Linearithmic time
  - O(n²) - Quadratic time
  - O(n³) - Cubic time
  - O(c^n) - Exponential time
- Generates visualization of data with curve fitting
- Provides interpretation of results with recommendations

## Usage

```bash
python3 algorithm_complexity.py --load-sizes [SIZES] --times [TIMES] [options]
```

### Command Line Options

| Option | Description | Required |
|--------|-------------|---------|
| `--load-sizes` | List of load sizes or input sizes | Yes |
| `--times` | List of execution times corresponding to load sizes | Yes |
| `--output-dir` | Output directory for plots (default: current directory) | No |
| `--show-plot` | Display the plot (default: False) | No |

## Example
```bash
python3 algorithm_complexity.py --load-sizes 10 20 50 100 200 500 --times 0.1 0.15 0.3 0.6 1.1 2.7 --output-dir ./results
```

This command will analyze the algorithmic complexity of a system with the provided load sizes and execution times, then save a visualization to the specified output directory.

## Output
The script produces:
1. A plot showing the original data points and fitted curves for each complexity model
2. Analysis output including:
   - Best fitting model
   - Confidence in the model fit
   - Explanation of the complexity pattern
   - Performance implications
   - Optimization recommendations

## API Usage
The module can also be imported and used programmatically:

```python
from algorithm_complexity import analyze_algorithm_complexity, plot_algorithm_complexity, interpret_algorithm_complexity

load_sizes = [10, 20, 50, 100, 200]
execution_times = [0.1, 0.15, 0.3, 0.6, 1.2]

# Analyze complexity
analysis = analyze_algorithm_complexity(load_sizes, execution_times)

# Generate plot
plot_path = plot_algorithm_complexity(load_sizes, execution_times, analysis, output_dir='./results')

# Interpret results
interpretation = interpret_algorithm_complexity(analysis)
```
