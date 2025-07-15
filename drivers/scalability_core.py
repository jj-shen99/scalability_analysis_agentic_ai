#!/usr/bin/env python3
"""
Core Scalability Analysis Module

This module provides core functionality for scalability analysis including:
- Basic JTL file parsing and performance metrics calculation
- Data structures for storing and manipulating performance data
- Common utilities used by other scalability modules

Part of the modular scalability analysis framework
"""

import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
import xml.etree.ElementTree as ET


def analyze_jtl(file_path):
    """
    Analyzes a single JTL file (XML or CSV format) and returns key performance metrics.
    
    Args:
        file_path (str): Path to the JTL file to analyze
        
    Returns:
        dict: Performance metrics or None if file couldn't be parsed
    """
    print(f"Analyzing JTL file: {file_path}")
    
    # Determine file format based on extension
    if file_path.lower().endswith('.xml') or file_path.lower().endswith('.jtl'):
        # Try XML parsing first
        try:
            return _analyze_xml_jtl(file_path)
        except ET.ParseError as e:
            # If XML parsing fails, try as CSV
            print(f"XML parsing failed, trying as CSV: {e}")
            return _analyze_csv_jtl(file_path)
    else:
        # Default to CSV for other extensions
        return _analyze_csv_jtl(file_path)


def _analyze_xml_jtl(file_path):
    """Parse JTL in XML format and extract metrics"""
    samples = []
    try:
        # Use iterparse for robust parsing of potentially large or malformed files
        for event, elem in ET.iterparse(file_path, events=('end',)):
            if elem.tag == 'sample' or elem.tag == 'httpSample':
                try:
                    samples.append({
                        'timeStamp': int(elem.get('ts')),
                        'elapsed': int(elem.get('t')),
                        'success': elem.get('s') == 'true',
                        'label': elem.get('lb', 'N/A'),
                        'responseCode': elem.get('rc', 'N/A')
                    })
                except (TypeError, ValueError):
                    # This sample has missing or invalid attributes
                    pass
                finally:
                    # Free up memory
                    elem.clear()
    except ET.ParseError as e:
        print(f"Warning: XML parsing error in {file_path}: {e}")
        raise

    return _calculate_metrics_from_samples(samples, file_path)


def _analyze_csv_jtl(file_path):
    """Parse JTL in CSV format and extract metrics"""
    try:
        # Try to read with pandas, inferring appropriate columns
        df = pd.read_csv(file_path, low_memory=False)
        
        # Convert to samples list for consistent processing
        samples = []
        for _, row in df.iterrows():
            sample = {}
            # Map common CSV columns to our sample format
            if 'timeStamp' in df.columns:
                sample['timeStamp'] = int(row['timeStamp'])
            elif 'timestamp' in df.columns:
                sample['timeStamp'] = int(row['timestamp'])
                
            if 'elapsed' in df.columns:
                sample['elapsed'] = int(row['elapsed'])
            elif 'Latency' in df.columns:
                sample['elapsed'] = int(row['Latency'])
                
            if 'success' in df.columns:
                sample['success'] = row['success'] == True or row['success'] == 'true'
            elif 'success' in df.columns:
                sample['success'] = row['success'] == True or row['success'] == 'true'
                
            if 'label' in df.columns:
                sample['label'] = row['label']
            elif 'Label' in df.columns:
                sample['label'] = row['Label']
            else:
                sample['label'] = 'N/A'
                
            if 'responseCode' in df.columns:
                sample['responseCode'] = row['responseCode']
            elif 'responseCode' in df.columns:
                sample['responseCode'] = row['responseCode']
            else:
                sample['responseCode'] = 'N/A'
                
            samples.append(sample)
            
        return _calculate_metrics_from_samples(samples, file_path)
    
    except Exception as e:
        print(f"Error parsing CSV file {file_path}: {e}")
        return None


def _calculate_metrics_from_samples(samples, file_path):
    """Calculate performance metrics from a list of samples"""
    if not samples:
        print(f"Error: No valid sample data could be read from {file_path}")
        return None

    # Convert to dataframe for easier analysis
    df = pd.DataFrame(samples)
    
    # Convert timestamp to datetime if it's not already
    if 'timeStamp' in df.columns:
        if pd.api.types.is_numeric_dtype(df['timeStamp']):
            df['timeStamp'] = pd.to_datetime(df['timeStamp'], unit='ms')

    # Calculate test duration
    if df.empty:
        return None
        
    # Calculate metrics differently based on whether we have timestamps
    if 'timeStamp' in df.columns:
        # Time-based metrics
        start_time = df['timeStamp'].min()
        end_time = df['timeStamp'].max()
        duration_seconds = (end_time - start_time).total_seconds()
        if duration_seconds <= 0:
            duration_seconds = sum(df['elapsed']) / 1000  # fallback to sum of elapsed times
    else:
        # No timestamps, use sum of elapsed times
        duration_seconds = sum(df['elapsed']) / 1000

    # Calculate basic metrics
    total_requests = len(df)
    throughput = total_requests / duration_seconds if duration_seconds > 0 else 0
    avg_response_time = df['elapsed'].mean()
    
    # Error statistics
    if 'success' in df.columns:
        error_count = df[df['success'] == False].shape[0]
    else:
        error_count = 0
    
    error_percentage = (error_count / total_requests) * 100 if total_requests > 0 else 0

    # Calculate percentiles for response times
    percentiles = {}
    if not df.empty:
        for p in [50, 90, 95, 99]:
            percentiles[f"p{p}"] = float(np.percentile(df['elapsed'], p))

    return {
        'total_requests': total_requests,
        'duration_seconds': duration_seconds,
        'throughput': throughput,  # requests/second
        'avg_response_time': avg_response_time,  # milliseconds
        'error_percentage': error_percentage,
        'percentiles': percentiles
    }


def create_output_dir(base_dir=None, timestamp_format="%Y%m%d_%H%M%S", custom_name=None, resource_levels=None):
    """
    Create an output directory for analysis results with a meaningful name
    
    Args:
        base_dir (str): Base directory for output, defaults to "sample_analysis_results"
        timestamp_format (str): Format for timestamp directory
        custom_name (str, optional): Custom name for the directory
        resource_levels (list, optional): Resource levels used in the analysis
        
    Returns:
        str: Path to created directory
    """
    if base_dir is None:
        base_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "sample_analysis_results"
        )
    
    # Create a descriptive directory name
    timestamp = datetime.now().strftime(timestamp_format)
    
    if custom_name:        
        # Use custom name if provided
        dir_name = f"{custom_name}_{timestamp}"
    elif resource_levels:
        # Create name based on resource levels if available
        levels_str = "_".join(str(level) for level in resource_levels)
        dir_name = f"ScalabilityAnalysis_Nodes{levels_str}_{timestamp}"
    else:
        # Default to timestamp-based name
        dir_name = f"ScalabilityAnalysis_{timestamp}"
    
    output_dir = os.path.join(base_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir


class JSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles non-serializable objects
    """
    def default(self, obj):
        # Handle function objects
        if callable(obj):
            return f"<function {obj.__name__}>"
        # Handle numpy arrays
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # Handle numpy numeric types
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                              np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        # Handle sets
        elif isinstance(obj, set):
            return list(obj)
        # Let the base class handle other types or raise a TypeError
        return super().default(obj)


def save_results_json(results, output_dir):
    """
    Save analysis results to a JSON file
    
    Args:
        results (dict): Analysis results to save
        output_dir (str): Directory to save the file
        
    Returns:
        str: Path to saved file
    """
    json_file = os.path.join(output_dir, "scalability_results.json")
    
    try:
        # Helper function to make results JSON serializable
        def make_serializable(obj):
            if callable(obj):
                return f"<function {obj.__name__}>"
            elif hasattr(obj, 'tolist') and callable(obj.tolist):
                return obj.tolist()
            elif hasattr(obj, 'item') and callable(obj.item):
                return obj.item()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            else:
                return str(obj)
        
        # Convert to JSON with custom handling for non-serializable objects
        with open(json_file, "w") as f:
            json.dump(results, f, indent=2, default=make_serializable)
    except Exception as e:
        print(f"Warning: Error saving full results: {e}")
        # Fallback to basic information only
        basic_info = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "files_analyzed": [r.get('file', 'unknown') for r in results.get('analysis_results', [])],
            "plots_generated": list(results.get('plot_paths', {}).keys()),
            "reports_generated": list(results.get('report_paths', {}).keys())
        }
        with open(json_file, "w") as f:
            json.dump(basic_info, f, indent=2)
    
    return json_file


# Main entry point for direct script execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Core Scalability Analysis Module')
    parser.add_argument('--jtl-file', type=str, required=True, help='JTL file to analyze')
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory if not specified
    output_dir = args.output_dir or create_output_dir()
    
    # Analyze JTL file
    metrics = analyze_jtl(args.jtl_file)
    
    if metrics:
        # Print basic metrics
        print(f"\nPerformance Metrics:")
        print(f"  Total Requests: {metrics['total_requests']}")
        print(f"  Duration: {metrics['duration_seconds']:.2f} seconds")
        print(f"  Throughput: {metrics['throughput']:.2f} req/sec")
        print(f"  Avg Response Time: {metrics['avg_response_time']:.2f} ms")
        print(f"  Error Rate: {metrics['error_percentage']:.2f}%")
        
        # Save results to JSON
        json_path = save_results_json({"metrics": metrics}, output_dir)
        print(f"\nDetailed results saved to: {json_path}")
    else:
        print("Analysis failed - no valid data found in JTL file")
