# Scalability Analyzer - Main Entry Point

## Overview
The Scalability Analyzer Main module serves as the primary entry point for the modular scalability analysis framework. It integrates all the components of the framework including JTL parsing, scalability models, visualization, and report generation in multiple formats.

## Features
- Centralized command-line interface for the complete scalability analysis workflow
- Integrates multiple analysis components:
  - Core JTL parsing and metrics calculation
  - Scalability model analysis (Amdahl's Law, Gustafson's Law, USL)
  - Algorithm complexity analysis
  - Load scalability analysis 
  - Visualization generation
  - Reporting in multiple formats (Markdown, HTML, DOCX)
- Customizable analysis options with extensive command-line parameters
- Supports batch processing of multiple test files

## Usage

```bash
python3 scalability_analyzer_main.py --files [FILES] --levels [LEVELS] [options]
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--files` | List of JTL file paths to analyze | Required |
| `--levels` | Resource levels for each file (e.g., node counts) | Required |
| `--output-dir` | Directory to save results | sample_analysis_results |
| `--formats` | Report formats to generate (md, html, docx) | md,html |
| `--show-plots` | Display plots during analysis | False |
| `--algorithm-complexity` | Perform algorithm complexity analysis | False |
| `--load-scalability` | Perform load scalability analysis | False |
| `--load-levels` | Load levels for load scalability analysis | Same as resource levels |
| `--max-projection` | Maximum resource level to project for theoretical models | 32 |
| `--advanced-visualizations` | Generate advanced visualization plots | False |

## Example
```bash
python3 scalability_analyzer_main.py \
  --files test_2node.jtl test_4node.jtl test_8node.jtl \
  --levels 2 4 8 \
  --output-dir ./scalability_results \
  --formats md,html,docx \
  --algorithm-complexity \
  --load-scalability \
  --advanced-visualizations
```

## Output
The script generates a timestamped directory with comprehensive analysis results:

1. **Performance Summary**:
   - Metrics for each test file (throughput, response time, error rates)
   - Resource scaling efficiency calculations

2. **Scalability Analysis**:
   - Fitted parameters for scalability models
   - Interpretation of scaling behavior
   - Optimization suggestions

3. **Optional Analyses**:
   - Algorithm complexity classification
   - Load scalability patterns and saturation points

4. **Visualizations**:
   - Resource vs. throughput/response time plots
   - Speedup vs. resource level
   - Model comparison plots
   - Theoretical projections
   - Optional advanced visualizations

5. **Reports**:
   - Complete analysis reports in selected formats
   - JSON export of all analysis data

## Integration
This module integrates the following components:
- scalability_core
- scalability_models
- algorithm_complexity
- load_scalability
- scalability_visualization_basic
- scalability_reporting_md, scalability_reporting_html, scalability_reporting_docx
