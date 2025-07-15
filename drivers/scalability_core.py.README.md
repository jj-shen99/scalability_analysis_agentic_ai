# Core Scalability Analysis Module

## Overview
This module provides core functionality for the scalability analysis framework including JTL file parsing, performance metrics calculation, data structures for storing results, and common utilities used by other modules in the framework.

## Features
- Robust JTL file parsing for both XML and CSV formats
- Automatic format detection and fallback mechanisms
- Comprehensive performance metrics calculation
- Consistent data structures for analysis results
- JSON serialization utilities for saving analysis results
- Timestamp-based directory structure for organizing outputs

## Usage

```bash
python3 scalability_core.py --jtl-file [JTL_FILE] [options]
```

### Command Line Options

| Option | Description | Required |
|--------|-------------|---------|
| `--jtl-file` | JTL file to analyze | Yes |
| `--output-dir` | Output directory for results (default: auto-generated) | No |

## Example
```bash
python3 scalability_core.py --jtl-file test_results.jtl --output-dir ./analysis_results
```

## Output
When run directly, the script outputs:
1. Basic performance metrics to the console:
   - Total requests
   - Test duration
   - Throughput (requests/second)
   - Average response time (milliseconds)
   - Error rate percentage

2. Detailed JSON results file containing all extracted metrics

## API Functions

### `analyze_jtl(file_path)`
Analyzes a JTL file and returns key performance metrics.
- **Parameters:** `file_path` - Path to the JTL file
- **Returns:** Dictionary with performance metrics or None if parsing fails

### `create_output_dir(base_dir, timestamp_format, custom_name, resource_levels)`
Creates an output directory for analysis results with a meaningful name.
- **Parameters:** 
  - `base_dir` - Base directory (default: "sample_analysis_results")
  - `timestamp_format` - Format for timestamp directory (default: "%Y%m%d_%H%M%S")
  - `custom_name` - Optional custom name prefix
  - `resource_levels` - Optional list of resource levels for naming
- **Returns:** Path to created directory

### `save_results_json(results, output_dir)`
Saves analysis results to a JSON file with special handling for non-serializable objects.
- **Parameters:**
  - `results` - Analysis results to save
  - `output_dir` - Directory to save the file
- **Returns:** Path to saved JSON file

## Integration
This module is used by other components in the scalability analysis framework as the foundation for JTL parsing and metrics extraction. It's typically imported by higher-level modules rather than used directly.
