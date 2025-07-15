# Scalability Reporting Module - Markdown Generator

## Overview
This module generates comprehensive Markdown reports from scalability analysis results. It converts complex performance data and analysis results into a well-structured, human-readable document that can be easily viewed or converted to other formats.

## Features
- Creates clean, well-formatted Markdown documentation
- Organizes scalability analysis results into logical sections
- Presents performance metrics in easy-to-read tables
- Summarizes scalability model parameters and interpretations
- Highlights key findings and optimization recommendations
- Supports advanced analysis results including algorithm complexity and load scalability
- Provides lightweight, portable documentation that can be committed to version control

## Usage
This module is typically used as a component of the scalability analysis framework rather than directly. However, it can be used independently:

```python
from scalability_reporting_md import generate_markdown_report

# Generate markdown report from analysis results
report_path = generate_markdown_report(
    analysis_results=results,
    output_dir='./reports'
)
```

### Command Line Usage
```bash
python3 scalability_reporting_md.py --files file1.jtl file2.jtl --levels 2 4 --output-dir ./reports
```

### Command Line Options

| Option | Description | Required |
|--------|-------------|---------|
| `--files` | JTL files to analyze | Yes |
| `--levels` | Resource levels corresponding to files | Yes |
| `--output-dir` | Output directory for report | No |

## Report Structure
The generated Markdown report includes:

1. **Title and timestamp** - Report title and generation date
2. **Executive Summary** - Overview of the analysis scope and key findings
3. **Detailed Performance Metrics** - Tabular presentation of performance data
4. **Scalability Analysis** - Results from scalability model fitting
   - Amdahl's Law analysis
   - Gustafson's Law analysis
   - Universal Scalability Law analysis
5. **Advanced Analysis** (if available)
   - Algorithm complexity analysis
   - Load scalability analysis
6. **Optimization Suggestions** - System-specific optimization recommendations

## Dependencies
- scalability_core - For JTL parsing when run directly

## Integration
This module is used by the main scalability analyzer to generate Markdown format reports as part of the comprehensive analysis workflow. The Markdown output can also be converted to other formats using tools like the included HTML converter.
