# Modular Scalability Analyzer Framework

This framework provides comprehensive scalability analysis for performance testing, with functionality to analyze JMeter JTL files and generate detailed reports on scalability characteristics.

## Key Features

- **JTL File Parsing**: Analyze JMeter JTL files in both XML and CSV formats
- **Scalability Model Analysis**: Implements three key scalability laws (Amdahl's, Gustafson's, Universal Scalability Law)
- **Advanced Visualization**: Generate insightful plots for scalability analysis
- **Comprehensive Reporting**: Create reports in multiple formats (Markdown, HTML, DOCX)
- **Theoretical Projections**: Generate theoretical projections even with limited data points
- **Comparative Analysis**: Compare performance across different node configurations

## Architecture

The framework is organized into modular components:

1. **Core Module** (`scalability_core.py`): Provides basic JTL file parsing and metrics calculation
2. **Models Module** (`scalability_models.py`): Implements scalability laws and mathematical models
3. **Visualization Modules**:
   - `scalability_visualization_basic.py`: Basic plotting capabilities
4. **Reporting Modules**:
   - `scalability_reporting_md.py`: Markdown report generation
   - `scalability_reporting_html.py`: HTML report generation
   - `scalability_reporting_docx.py`: DOCX report generation
5. **Main Module** (`scalability_analyzer_main.py`): Integrates all components into a unified workflow

## Usage

### Basic Usage

```bash
python scalability_analyzer_main.py --files file1.jtl file2.jtl --levels 2 4 --output-dir results
```

### Advanced Options

```bash
python scalability_analyzer_main.py \
  --files file1.jtl file2.jtl file3.jtl \
  --levels 2 4 8 \
  --output-dir results \
  --formats md html docx \
  --sla 200 \
  --configs "2-node" "4-node" "8-node" \
  --comparative
```

### Parameters

- `--files`: JTL files to analyze
- `--levels`: Resource levels corresponding to each file (e.g., node counts)
- `--output-dir`: Directory to save results (optional)
- `--formats`: Report formats to generate (optional, defaults to all)
- `--sla`: SLA threshold for response time in ms (optional)
- `--show-plots`: Display plots during analysis (optional)
- `--comparative`: Generate comparative analysis between configurations (optional)
- `--configs`: Configuration names for each JTL file (optional)

## Report Content

The generated reports include:

- **Executive Summary**: Key findings and metrics
- **Performance Metrics**: Detailed throughput, response time, and error rate analysis
- **Scalability Metrics**: Analysis of speedup and efficiency
- **Advanced Scalability Analysis**: Model parameters and interpretations
- **Visual Analysis**: Interactive plots and visualizations
- **Theoretical Projections**: Projected performance with additional resources
- **Optimization Suggestions**: Recommendations for improving scalability

## Examples

### Analyzing Basic Scalability

For basic scalability analysis with two node configurations:

```bash
python scalability_analyzer_main.py \
  --files /path/to/2node.jtl /path/to/4node.jtl \
  --levels 2 4
```

### Comprehensive Analysis with SLA

For more comprehensive analysis including SLA threshold:

```bash
python scalability_analyzer_main.py \
  --files /path/to/2node.jtl /path/to/4node.jtl /path/to/8node.jtl \
  --levels 2 4 8 \
  --sla 500 \
  --configs "2-node" "4-node" "8-node" \
  --comparative
```

## Contributing

To extend this framework:

1. Add new visualization types to `scalability_visualization_basic.py`
2. Implement additional scalability models in `scalability_models.py`
3. Enhance reporting capabilities in the reporting modules

## Dependencies

- Python 3.6+
- pandas
- numpy
- matplotlib
- scipy
- docx
- markdown2
- seaborn

## License

This software is part of the Performance Testing Framework.
