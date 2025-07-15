# Load Scalability Analysis Module

## Overview
This module analyzes how a system's performance changes under different load levels while resources remain constant. It helps identify saturation points, capacity limits, and performance degradation patterns by analyzing throughput and response time metrics across varying load levels.

## Features
- Analyzes performance metrics across different load levels
- Identifies throughput saturation points using the elbow method
- Detects response time inflection points where performance degrades
- Applies Little's Law to validate system behavior
- Generates visualization of load scalability data
- Provides interpretation of results with actionable insights and recommendations

## Usage

```bash
python3 load_scalability.py --load-levels [LEVELS] --throughputs [VALUES] --response-times [VALUES] [options]
```

### Command Line Options

| Option | Description | Required |
|--------|-------------|---------|
| `--load-levels` | List of load levels (e.g., concurrent users or request rates) | Yes |
| `--throughputs` | List of corresponding throughput values | Yes |
| `--response-times` | List of corresponding response time values | Yes |
| `--output-dir` | Output directory for plots (default: current directory) | No |
| `--show-plot` | Display the plots (default: False) | No |

## Example
```bash
python3 load_scalability.py --load-levels 10 20 50 100 200 500 --throughputs 100 190 450 800 1000 950 --response-times 50 55 60 90 150 350 --output-dir ./results
```

This command will analyze the system's load scalability based on the provided metrics and save visualization plots to the specified output directory.

## Output
The script produces:
1. A throughput vs. load plot showing the saturation point
2. A response time vs. load plot showing the inflection point
3. A combined plot showing both metrics and optimal operating points
4. Analysis output including:
   - Saturation point (where throughput levels off)
   - Optimal load level (best balance of throughput and response time)
   - Insights into system behavior
   - Recommendations for capacity planning and optimization

## API Usage
The module can also be imported and used programmatically:

```python
from load_scalability import analyze_load_scalability, plot_load_scalability, interpret_load_scalability

load_levels = [10, 20, 50, 100, 200, 500]
throughputs = [100, 190, 450, 800, 1000, 950]
response_times = [50, 55, 60, 90, 150, 350]

# Analyze load scalability
analysis = analyze_load_scalability(load_levels, throughputs, response_times)

# Generate plots
plot_paths = plot_load_scalability(
    load_levels, throughputs, response_times, 
    analysis, output_dir='./results'
)

# Interpret results
interpretation = interpret_load_scalability(analysis)
```
