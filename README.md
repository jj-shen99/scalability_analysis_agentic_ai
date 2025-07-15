# Scalability Analysis Framework for Agentic AI

A comprehensive framework for analyzing and visualizing the scalability characteristics of performance testing data, with a focus on agentic AI systems.

## Overview

This framework provides tools for analyzing JMeter JTL files and other performance testing data to evaluate scalability characteristics across different resource configurations. It implements multiple scalability laws (Amdahl's, Gustafson's, Universal Scalability Law) to provide insights into system performance under varying loads and resource allocations.

## Directory Structure

- **`/scalability_analysis/`**: Core framework modules
  - Scalability models implementation
  - JTL file parsing and analysis
  - Visualization and reporting capabilities
  - Main execution module
- **`/sample_analysis_results/`**: Example analysis outputs and reports
  - Organized by test runs and model types
- **`/sample_synthetic_jtl/`**: Sample JTL files for testing
- **`/workflows/`**: Analysis workflow definitions
- **`/drivers/`**: Python scripts for specific analysis tasks
- **`/docs/`**: Comprehensive documentation
  - User guide
  - Technical summary

## Key Features

- **JTL File Analysis**: Process JMeter JTL files in both XML and CSV formats
- **Multiple Scalability Models**:
  - Amdahl's Law for fixed-workload scaling
  - Gustafson's Law for scaled-workload analysis
  - Universal Scalability Law (USL) for comprehensive modeling
- **Advanced Visualization**: Generate insightful plots for scalability analysis
- **Multi-format Reporting**: Create reports in Markdown, HTML, and DOCX formats
- **Theoretical Projections**: Generate performance projections with limited data points
- **Comparative Analysis**: Compare performance across different node configurations
- **Algorithm Complexity Analysis**: Evaluate algorithmic scaling characteristics
- **Load Scalability Analysis**: Analyze system behavior under different load conditions

## Installation

1. Ensure you have Python 3.6+ installed
2. Clone this repository
3. Install dependencies:

```bash
pip install pandas numpy matplotlib scipy python-docx markdown2 seaborn
```

## Usage

### Basic Usage

```bash
python3 drivers/scalability_analyzer_main.py --files file1.jtl file2.jtl --levels 2 4 --output-dir sample_analysis_results/new_analysis
```

### Advanced Options

```bash
python3 drivers/scalability_analyzer_main.py \
  --files sample_synthetic_jtl/file1.jtl sample_synthetic_jtl/file2.jtl sample_synthetic_jtl/file3.jtl \
  --levels 2 4 8 \
  --output-dir sample_analysis_results/comprehensive_analysis \
  --formats md html docx \
  --sla 200 \
  --configs "2-node" "4-node" "8-node" \
  --comparative
```

### Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--files` | JTL files to analyze (space-separated) | `--files file1.jtl file2.jtl` |
| `--levels` | Resource levels for each file | `--levels 2 4 8` |
| `--output-dir` | Directory to save results | `--output-dir results/test1` |
| `--formats` | Report formats to generate | `--formats md html docx` |
| `--sla` | SLA threshold for response time (ms) | `--sla 200` |
| `--show-plots` | Display plots during analysis | `--show-plots` |
| `--comparative` | Generate comparative analysis | `--comparative` |
| `--configs` | Configuration names for each file | `--configs "2-node" "4-node"` |
| `--algorithm-complexity` | Analyze algorithm complexity | `--algorithm-complexity` |
| `--load-scalability` | Analyze load scalability | `--load-scalability` |
| `--load-levels` | Load levels for load scalability | `--load-levels 10 20 30` |
| `--workflow` | Path to workflow definition file | `--workflow workflows/custom.md` |

## Example Workflows

See the `/workflows` directory for example analysis workflows. For instance:

```bash
# Follow the workflow defined in analysis_synthetic_modeltest.md
python3 drivers/scalability_analyzer_main.py \
  --workflow workflows/analysis_synthetic_modeltest.md
```

## Report Content

The generated reports include:

- **Executive Summary**: Key findings and metrics
- **Performance Metrics**: Detailed throughput, response time, and error rate analysis
- **Scalability Metrics**: Analysis of speedup and efficiency
- **Advanced Scalability Analysis**: Model parameters and interpretations
- **Visual Analysis**: Interactive plots and visualizations
- **Theoretical Projections**: Projected performance with additional resources
- **Optimization Suggestions**: Recommendations for improving scalability

## Documentation

Comprehensive documentation is available in the `/docs` directory:

- **[User Guide](docs/user_guide.md)**: Complete guide for using the framework, including:
  - Installation and setup
  - Input data requirements
  - Basic and advanced usage
  - Understanding scalability models
  - Interpreting results
  - Troubleshooting

- **[Technical Summary](docs/technical_summary.md)**: Technical overview of the framework, including:
  - Architecture design
  - Module interactions
  - Data flow diagrams
  - Implementation details
  - Performance considerations
  - Future development roadmap

## Dependencies

- Python 3.6+
- pandas
- numpy
- matplotlib
- scipy
- python-docx
- markdown2
- seaborn

## Contributing

To extend this framework:

1. Add new visualization types to `scalability_visualization_basic.py`
2. Implement additional scalability models in `scalability_models.py`
3. Enhance reporting capabilities in the reporting modules
4. Create new analysis workflows in the `/workflows` directory

## License

This software is part of the Performance Testing Framework.
