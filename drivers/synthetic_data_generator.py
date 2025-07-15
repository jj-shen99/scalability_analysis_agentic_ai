#!/usr/bin/env python3
"""
Synthetic Data Generator for Scalability Analysis

This module generates synthetic JTL data files with realistic performance patterns
for testing the scalability analysis framework with different node configurations
and scalability characteristics.

Author: JJ Shen
"""

import os
import csv
import random
import numpy as np
import argparse
from datetime import datetime, timedelta

# Default settings for synthetic data
DEFAULT_SETTINGS = {
    "transaction_count": 5000,     # Number of transactions per file
    "base_throughput": 100.0,      # Base throughput for 1 node (req/sec)
    "base_response_time": 200.0,   # Base response time for 1 node (ms)
    "error_rate": 0.01,            # Error rate (fraction of requests)
    "time_variance": 0.2,          # Response time variance (fraction of mean)
    "scalability_pattern": "amdahl",  # Scalability pattern (amdahl, gustafson, usl, sublinear)
    "amdahl_p": 0.8,               # Amdahl's Law parallelizable fraction
    "usl_sigma": 0.1,              # USL contention factor
    "usl_kappa": 0.02,             # USL coherency delay factor
    "sublinear_factor": 0.8,       # Sublinear scaling factor (< 1)
    "start_timestamp": "2025-07-08T00:00:00",  # Start timestamp for transactions
    "node_counts": [2, 4, 8]       # Node configurations to generate data for
}


def generate_timestamps(count, start_time, mean_rate):
    """
    Generate timestamps for transactions based on mean request rate
    
    Args:
        count (int): Number of timestamps to generate
        start_time (datetime): Start time for the first transaction
        mean_rate (float): Mean request rate in requests per second
        
    Returns:
        list: List of epoch timestamps as integers (milliseconds since epoch)
    """
    # Convert mean rate to mean time between requests in seconds
    mean_interval = 1.0 / mean_rate
    
    # Generate random intervals with exponential distribution (typical for arrival processes)
    intervals = np.random.exponential(mean_interval, count)
    
    # Calculate cumulative times and add to start time
    cum_times = np.cumsum(intervals)
    timestamps = [start_time + timedelta(seconds=t) for t in cum_times]
    
    # Convert to epoch milliseconds as integers (JMeter format)
    return [(int(t.timestamp() * 1000)) for t in timestamps]


def calculate_scaling_factor(node_count, settings):
    """
    Calculate scaling factor for throughput based on the selected scalability pattern
    
    Args:
        node_count (int): Number of nodes
        settings (dict): Settings for synthetic data generation
        
    Returns:
        float: Scaling factor for throughput
    """
    pattern = settings["scalability_pattern"]
    
    if pattern == "linear":
        # Perfect linear scaling (ideal case)
        return node_count
    
    elif pattern == "amdahl":
        # Amdahl's Law scaling
        p = settings["amdahl_p"]  # Parallelizable fraction
        return 1.0 / ((1.0 - p) + p / node_count)
    
    elif pattern == "gustafson":
        # Gustafson's Law scaling
        p = settings["amdahl_p"]  # Parallelizable fraction
        return (1.0 - p) + p * node_count
    
    elif pattern == "usl":
        # Universal Scalability Law
        sigma = settings["usl_sigma"]   # Contention factor
        kappa = settings["usl_kappa"]   # Coherency delay factor
        return node_count / (1 + sigma * (node_count - 1) + kappa * node_count * (node_count - 1))
    
    elif pattern == "sublinear":
        # Sublinear scaling (throughput ~ nodes^factor)
        factor = settings["sublinear_factor"]
        return node_count ** factor
    
    else:
        # Default to linear scaling
        return node_count


def generate_jtl_data(node_count, output_file, settings):
    """
    Generate synthetic JTL data for a specific node configuration
    
    Args:
        node_count (int): Number of nodes
        output_file (str): Path to output JTL file
        settings (dict): Settings for synthetic data generation
        
    Returns:
        dict: Statistics for generated data
    """
    # Calculate scaling factor
    scaling_factor = calculate_scaling_factor(node_count, settings)
    
    # Calculate expected throughput and response time
    throughput = settings["base_throughput"] * scaling_factor
    response_time = settings["base_response_time"] / scaling_factor
    
    # Calculate timestamps based on throughput
    count = settings["transaction_count"]
    start_time = datetime.strptime(settings["start_timestamp"], "%Y-%m-%dT%H:%M:%S")
    timestamps = generate_timestamps(count, start_time, throughput)
    
    # Generate transaction labels
    labels = ["Transaction-" + str(i % 5 + 1) for i in range(count)]
    
    # Generate response times with variance
    rt_variance = response_time * settings["time_variance"]
    response_times = np.random.normal(response_time, rt_variance, count)
    response_times = np.maximum(response_times, 1.0)  # Ensure positive values
    
    # Generate response codes (some errors based on error rate)
    error_count = int(count * settings["error_rate"])
    response_codes = ["200"] * (count - error_count) + ["500"] * error_count
    random.shuffle(response_codes)
    
    # Create CSV data rows
    rows = []
    for i in range(count):
        # Common fields in JTL file exactly matching JMeter format
        # Note: timeStamp must be an integer (milliseconds since epoch)
        elapsed = int(response_times[i])
        row = {
            "timeStamp": str(timestamps[i]),
            "elapsed": str(elapsed),
            "label": labels[i],
            "responseCode": response_codes[i],
            "responseMessage": "OK" if response_codes[i] == "200" else "Internal Server Error",
            "threadName": f"Thread Group {random.randint(1, node_count)}-{random.randint(1, 5)}",
            "dataType": "text",
            "success": "true" if response_codes[i] == "200" else "false",
            "failureMessage": "" if response_codes[i] == "200" else "Test failed",
            "bytes": str(random.randint(500, 5000)),
            "sentBytes": str(random.randint(100, 500)),
            "grpThreads": str(node_count * 5),
            "allThreads": str(node_count * 5),
            "Latency": str(int(elapsed * 0.8)),
            "IdleTime": "0",
            "Connect": str(int(elapsed * 0.1))
        }
        rows.append(row)
    
    # Write to CSV file
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, 'w', newline='') as f:
        fieldnames = rows[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    # Calculate statistics
    success_times = [float(row["elapsed"]) for row in rows if row["success"] == "true"]
    error_times = [float(row["elapsed"]) for row in rows if row["success"] == "false"]
    
    success_count = len(success_times)
    error_count = len(error_times)
    
    stats = {
        "node_count": node_count,
        "transaction_count": count,
        "success_count": success_count,
        "error_count": error_count,
        "error_rate": error_count / count if count > 0 else 0,
        "avg_response_time": np.mean(success_times) if success_times else 0,
        "min_response_time": np.min(success_times) if success_times else 0,
        "max_response_time": np.max(success_times) if success_times else 0,
        "throughput": throughput,
        "expected_scaling_factor": scaling_factor
    }
    
    return stats


def generate_dataset(settings, output_dir):
    """
    Generate a complete dataset with multiple node configurations
    
    Args:
        settings (dict): Settings for synthetic data generation
        output_dir (str): Directory to save generated files
        
    Returns:
        list: Statistics for all generated files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    dataset_stats = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate JTL files for each node count
    for node_count in settings["node_counts"]:
        output_file = os.path.join(
            output_dir, f"synthetic_{settings['scalability_pattern']}_{node_count}node_{timestamp}.jtl"
        )
        
        print(f"Generating {node_count}-node data with {settings['scalability_pattern']} pattern...")
        stats = generate_jtl_data(node_count, output_file, settings)
        stats["file"] = output_file
        dataset_stats.append(stats)
        print(f"  - Created {output_file}")
        print(f"  - Throughput: {stats['throughput']:.2f} req/s, Avg Response Time: {stats['avg_response_time']:.2f} ms")
    
    # Create a summary file
    summary_file = os.path.join(output_dir, f"synthetic_dataset_{settings['scalability_pattern']}_{timestamp}.csv")
    with open(summary_file, 'w', newline='') as f:
        fieldnames = dataset_stats[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset_stats)
    
    print(f"\nSummary saved to: {summary_file}")
    print("\nDataset Statistics:")
    print(f"{'Nodes':<6} {'Throughput':<12} {'Resp Time':<12} {'Scaling Factor':<15}")
    print("-" * 45)
    for stats in dataset_stats:
        print(f"{stats['node_count']:<6} {stats['throughput']:<12.2f} {stats['avg_response_time']:<12.2f} {stats['expected_scaling_factor']:<15.3f}")
    
    return dataset_stats


def main():
    """Main function to generate synthetic datasets"""
    parser = argparse.ArgumentParser(description="Generate synthetic JTL data for scalability testing")
    
    parser.add_argument("--pattern", choices=["linear", "amdahl", "gustafson", "usl", "sublinear"],
                       default="amdahl", help="Scalability pattern to simulate")
    parser.add_argument("--nodes", nargs="+", type=int, default=[2, 3, 4, 8],
                       help="Node configurations to generate data for")
    parser.add_argument("--transactions", type=int, default=5000,
                       help="Number of transactions per file")
    parser.add_argument("--output-dir", type=str,
                       default="/Users/jianjun.shen/1___PerformanceTests/perf-gcs/AdvancedTrainning/jtl/synthetic",
                       help="Directory to save generated files")
    parser.add_argument("--amdahl-p", type=float, default=0.8,
                       help="Amdahl's Law parallelizable fraction (0-1)")
    parser.add_argument("--usl-sigma", type=float, default=0.1,
                       help="USL contention factor")
    parser.add_argument("--usl-kappa", type=float, default=0.02,
                       help="USL coherency delay factor")
    parser.add_argument("--sublinear-factor", type=float, default=0.8,
                       help="Sublinear scaling factor (0-1)")
    
    args = parser.parse_args()
    
    # Update settings from command line arguments
    settings = DEFAULT_SETTINGS.copy()
    settings["scalability_pattern"] = args.pattern
    settings["node_counts"] = args.nodes
    settings["transaction_count"] = args.transactions
    settings["amdahl_p"] = args.amdahl_p
    settings["usl_sigma"] = args.usl_sigma
    settings["usl_kappa"] = args.usl_kappa
    settings["sublinear_factor"] = args.sublinear_factor
    
    # Generate dataset
    generate_dataset(settings, args.output_dir)


if __name__ == "__main__":
    main()
