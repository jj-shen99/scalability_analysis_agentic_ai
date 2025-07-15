# Scalability Reporting Module - HTML Generator

## Overview
This module generates interactive HTML reports from scalability analysis results. It creates web-based documentation with responsive styling, interactive visualizations, and comprehensive data presentation that can be viewed in any modern web browser.

## Features
- Produces clean, well-formatted HTML reports with modern CSS styling
- Creates responsive layouts that work on different screen sizes
- Embeds visualizations directly into the report
- Provides interactive elements for better data exploration
- Includes comprehensive sections for all analysis aspects:
  - Executive summary and key findings
  - Performance metrics tables
  - Scalability model analysis
  - Advanced analysis results (algorithm complexity, load scalability)
  - Visualizations with explanations
  - Conclusions and recommendations
- Offers a secondary utility to convert Markdown reports to styled HTML

## Usage
This module is typically used as a component of the scalability analysis framework rather than directly. However, it can be used independently:

```python
from scalability_reporting_html import generate_html_report, convert_markdown_to_html

# Generate HTML report directly from analysis results
report_path = generate_html_report(
    analysis_results=results,
    output_dir='./reports',
    plot_paths={'throughput': 'path/to/throughput_plot.png', ...}
)

# Or convert an existing Markdown report to HTML
html_path = convert_markdown_to_html(
    markdown_path='./reports/scalability_report.md',
    output_dir='./reports'
)
```

## Functions

### `generate_html_report(analysis_results, output_dir, plot_paths=None)`
Generates a complete HTML report from scalability analysis results.

**Parameters:**
- `analysis_results`: List of dictionaries containing analysis results
- `output_dir`: Directory to save the HTML report
- `plot_paths`: Optional dictionary mapping plot types to image file paths

**Returns:**
- Path to the generated HTML report

### `convert_markdown_to_html(markdown_path, output_dir)`
Converts a Markdown report to HTML with enhanced styling.

**Parameters:**
- `markdown_path`: Path to the Markdown report file
- `output_dir`: Directory to save the HTML report

**Returns:**
- Path to the generated HTML report

## Report Structure
The generated HTML report includes:

1. **Header and title** with generation timestamp
2. **Executive Summary** highlighting test scope and key metrics
3. **Performance Metrics** section with detailed data tables
4. **Scalability Analysis** showing model parameters and interpretations
5. **Visualizations** section with embedded plots and explanatory captions
6. **Advanced Analysis** (if available) showing algorithm complexity and load scalability results
7. **Conclusions** with efficiency analysis and recommendations

## Dependencies
- markdown2 - For converting Markdown to HTML when using the conversion utility
