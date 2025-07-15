# Synthetic Data Generator

## Overview
The Synthetic Data Generator creates JTL data files with realistic performance patterns for testing the scalability analysis framework. It can simulate different node configurations and scalability characteristics including linear scaling, Amdahl's Law, Gustafson's Law, Universal Scalability Law (USL), and sublinear scaling.

## Features
- Generates synthetic JTL files with configurable parameters
- Supports multiple scalability patterns (linear, Amdahl, Gustafson, USL, sublinear)
- Creates datasets with multiple node configurations
- Produces summary statistics for generated data

## Usage

```bash
python3 synthetic_data_generator.py [options]
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--pattern` | Scalability pattern to simulate (linear, amdahl, gustafson, usl, sublinear) | amdahl |
| `--nodes` | Node configurations to generate data for | [2, 3, 4, 8] |
| `--transactions` | Number of transactions per file | 5000 |
| `--output-dir` | Directory to save generated files | /Users/.../jtl/synthetic |
| `--amdahl-p` | Amdahl's Law parallelizable fraction (0-1) | 0.8 |
| `--usl-sigma` | USL contention factor | 0.1 |
| `--usl-kappa` | USL coherency delay factor | 0.02 |
| `--sublinear-factor` | Sublinear scaling factor (0-1) | 0.8 |

## Output
The script generates:
1. Multiple JTL files (one per node configuration) containing synthetic performance data
2. A summary CSV file with statistics for all generated configurations

## Example
```bash
python3 synthetic_data_generator.py --pattern usl --nodes 2 4 8 16 --transactions 10000
```

This command will generate synthetic data using the Universal Scalability Law with node counts of 2, 4, 8, and 16, each file containing 10,000 transactions.

## Output Format
The generated JTL files follow the standard JMeter format with the following columns:
- timestamp: Request timestamp in milliseconds since epoch
- elapsed: Response time in milliseconds
- label: Transaction label
- responseCode: HTTP response code
- success: Whether the request was successful
- bytes: Response size in bytes
- grpThreads: Number of active threads in group
- allThreads: Total number of active threads
