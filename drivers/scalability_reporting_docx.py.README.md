# Scalability Reporting Module - DOCX Generator

## Overview
This module provides functionality for generating professional Microsoft Word (DOCX) reports from scalability analysis results. It creates comprehensive, well-formatted documentation with enhanced styling, embedded visualizations, and detailed interpretation of scalability analysis results.

## Features
- Generates polished Microsoft Word documents with consistent styling and formatting
- Creates hierarchical document structure with proper headings and sections
- Embeds visualization plots with captions
- Provides executive summary and key findings sections
- Presents detailed performance metrics with tables
- Includes comprehensive scalability model interpretations
- Offers system-specific observations and recommendations

## Usage
This module is typically used as a component of the scalability analysis framework rather than directly. However, it can be used independently:

```python
from scalability_reporting_docx import generate_docx_report

# Generate report from analysis results
report_path = generate_docx_report(
    analysis_results=results,  
    output_dir='./reports',
    plot_paths={'throughput': 'path/to/throughput_plot.png', ...}
)
```

### Command Line Usage
```bash
python3 scalability_reporting_docx.py --files file1.jtl file2.jtl --levels 2 4 --output-dir ./reports --plot-dir ./plots
```

### Command Line Options

| Option | Description | Required |
|--------|-------------|---------|
| `--files` | JTL files to analyze | Yes |
| `--levels` | Resource levels corresponding to files | Yes |
| `--output-dir` | Output directory for report | No |
| `--plot-dir` | Directory containing plot images to include | No |

## Report Structure
The generated DOCX report includes:

1. **Title and metadata** - Report title, generation date, and document properties
2. **Executive Summary** - Overview of the analysis scope and system
3. **Key Findings** - Highlight of important metrics and observations
4. **Performance Metrics** - Detailed tables of performance data
5. **Scalability Analysis** - Results from scalability model fitting
   - Amdahl's Law analysis
   - Gustafson's Law analysis
   - Universal Scalability Law analysis
6. **Visualization** - Embedded plots with captions (if plot paths provided)
7. **Conclusions and Recommendations** - System-specific observations and suggestions

## Dependencies
- python-docx - For creating and formatting Word documents
- scalability_core - For JTL parsing when run directly

## Integration
This module is used by the main scalability analyzer to generate DOCX format reports as part of the comprehensive analysis workflow.
