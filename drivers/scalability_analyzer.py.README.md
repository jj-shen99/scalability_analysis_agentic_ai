# Scalability Analyzer

## Overview
The Scalability Analyzer is a comprehensive tool for analyzing JMeter JTL files to evaluate system scalability across different resource levels. It applies multiple scalability models (Amdahl's Law, Gustafson's Law, and Universal Scalability Law) to predict system behavior and provides detailed reports with visualizations.

## Features
- Analyzes JMeter JTL files (XML format) for performance metrics
- Applies multiple scalability models:
  - Amdahl's Law - Evaluates fundamental speedup limits due to serial portions
  - Gustafson's Law - Models performance when workload grows with resources
  - Universal Scalability Law - Accounts for both contention and coherency delays
- Generates comprehensive reports in multiple formats:
  - Markdown
  - HTML
  - Microsoft Word (DOCX)
- Creates visualizations of actual vs. theoretical scaling
- Provides performance interpretations and optimization suggestions
- Calculates scaling efficiency and identifies potential bottlenecks

## Usage

```bash
python3 scalability_analyzer.py --files [FILES] --levels [LEVELS] [options]
```

### Command Line Options

| Option | Description | Required |
|--------|-------------|---------|
| `--files` | List of JTL file paths to analyze | Yes |
| `--levels` | Corresponding resource levels for each file (e.g., number of nodes) | Yes |
| `--output-dir` | Directory to save the report and plots (default: sample_analysis_results) | No |

## Example
```bash
python3 scalability_analyzer.py --files test_2node.jtl test_4node.jtl test_8node.jtl --levels 2 4 8 --output-dir ./scalability_results
```

This command analyzes three JTL files corresponding to 2, 4, and 8 node tests, and saves the reports to the specified output directory.

## Output
The script creates a timestamped subdirectory within the output directory containing:

1. **Reports**:
   - `scalability_report.md` - Markdown format
   - `scalability_report.html` - HTML format
   - `scalability_report.docx` - Microsoft Word format

2. **Visualizations**:
   - Actual vs. theoretical scaling curves
   - Resource vs. throughput plot
   - Resource vs. response time plot
   - Model comparisons and projections

3. **Analysis**:
   - Fitted parameters for each scalability model
   - Performance interpretations
   - Scaling efficiency calculations
   - Optimization suggestions

## Dependencies
- pandas
- matplotlib
- numpy
- python-docx
- markdown2
- scalability_models (internal module)

## Integration
This script is typically used in conjunction with other scalability analysis modules in the framework. It relies on the `scalability_models` module for the underlying mathematical models.
