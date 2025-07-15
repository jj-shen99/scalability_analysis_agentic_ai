#!/usr/bin/env python3
"""
Scalability Reporting Module - HTML Generator (Fixed Version)

This module provides functionality for generating HTML reports from
scalability analysis results with enhanced formatting and styling.

Part of the modular scalability analysis framework
"""

import os
from datetime import datetime
import markdown2
import json


def generate_html_report(analysis_results, output_dir, plot_paths=None):
    """
    Generate an HTML report from scalability analysis results
    
    Args:
        analysis_results (list): List of dictionaries with analysis results
        output_dir (str): Directory to save the report
        plot_paths (dict, optional): Dictionary of paths to plot images
        
    Returns:
        str: Path to the generated report
    """
    # Sort results by resource level for consistent reporting
    sorted_results = sorted(analysis_results, key=lambda x: x.get('resource_level', 0))
    
    # Extract resource levels and metrics for comparison
    resource_levels = [r.get('resource_level', 0) for r in sorted_results]
    throughputs = [r.get('metrics', {}).get('throughput', 0) for r in sorted_results]
    response_times = [r.get('metrics', {}).get('avg_response_time', 0) for r in sorted_results]
    error_percentages = [r.get('metrics', {}).get('error_percentage', 0) for r in sorted_results]
    
    # Calculate speedups relative to the baseline
    baseline_throughput = throughputs[0] if throughputs else 1
    speedups = [t / baseline_throughput for t in throughputs]
    
    # Calculate efficiency as speedup divided by relative resource count
    efficiencies = []
    for i, (level, speedup) in enumerate(zip(resource_levels, speedups)):
        if i == 0 or resource_levels[0] == 0:  # Avoid division by zero
            efficiencies.append(1.0)
        else:
            relative_resources = level / resource_levels[0]
            efficiencies.append(speedup / relative_resources if relative_resources > 0 else 0)

    # Get scalability analysis if available
    scalability_analysis = {}
    theoretical_projections = {}
    algorithm_complexity = {}
    load_scalability = {}
    
    for result in sorted_results:
        if 'scalability_analysis' in result:
            scalability_analysis = result['scalability_analysis']
            if 'theoretical_projections' in scalability_analysis:
                theoretical_projections = scalability_analysis['theoretical_projections']
            if 'advanced_analysis' in result:
                advanced_analysis = result['advanced_analysis']
                if 'algorithm_complexity' in advanced_analysis:
                    algorithm_complexity = advanced_analysis['algorithm_complexity']
                if 'load_scalability' in advanced_analysis:
                    load_scalability = advanced_analysis['load_scalability']
            break
    
    # Create CSS styling
    css_style = """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        .container {
            width: 90%;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #fff;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #2c3e50;
            margin-top: 30px;
        }
        h1 {
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f9f9f9;
        }
        .metrics-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            margin: 20px 0;
        }
        .metric-card {
            flex: 1;
            min-width: 200px;
            margin: 10px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
            margin: 10px 0;
        }
        .metric-label {
            font-size: 14px;
            color: #7f8c8d;
        }
        .plot-container {
            margin: 30px 0;
        }
        .plot-container img {
            width: 100%;
            max-width: 800px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .plot-description {
            margin-top: 10px;
            font-style: italic;
            color: #666;
        }
        .highlight {
            background-color: #fffacd;
            padding: 2px 5px;
            border-radius: 3px;
        }
        .good {
            color: #27ae60;
        }
        .warning {
            color: #e67e22;
        }
        .critical {
            color: #e74c3c;
        }
        .recommendation {
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }
        .card {
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .code {
            font-family: 'Courier New', Courier, monospace;
            background-color: #f5f5f5;
            padding: 2px 4px;
            border-radius: 3px;
        }
    """
    
    # Start building the HTML content - build it step by step to avoid unterminated string issues
    html_start = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scalability Analysis Report</title>
    <style>
{css_style}
    </style>
</head>
<body>
    <div class="container">
        <h1>Scalability Analysis Report</h1>
        <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <h2>Executive Summary</h2>
        <p>This report presents the results of scalability analysis performed across {len(resource_levels)} resource levels: 
        {', '.join([str(level) for level in resource_levels])}.</p>
"""
    
    # Initialize the full HTML content
    html_content = html_start
    
    # Add key metrics section
    html_content += """        <div class="metrics-container">"""
    
    # Add key metrics cards
    if throughputs:
        max_throughput = max(throughputs)
        max_throughput_level = resource_levels[throughputs.index(max_throughput)]
        html_content += f"""            <div class="metric-card">
                <div class="metric-label">Maximum Throughput</div>
                <div class="metric-value">{max_throughput:.2f} req/s</div>
                <div>at resource level {max_throughput_level}</div>
            </div>
"""
    
    if response_times:
        min_response_time = min(response_times)
        min_response_time_level = resource_levels[response_times.index(min_response_time)]
        html_content += f"""            <div class="metric-card">
                <div class="metric-label">Best Response Time</div>
                <div class="metric-value">{min_response_time:.2f} ms</div>
                <div>at resource level {min_response_time_level}</div>
            </div>
"""
    
    if len(speedups) > 1:
        max_speedup = max(speedups[1:], default=1)  # Skip baseline
        max_speedup_level = resource_levels[speedups.index(max_speedup)]
        efficiency = max_speedup / (max_speedup_level / resource_levels[0])
        efficiency_class = "good" if efficiency > 0.85 else "warning" if efficiency > 0.6 else "critical"
        
        html_content += f"""            <div class="metric-card">
                <div class="metric-label">Maximum Speedup</div>
                <div class="metric-value">{max_speedup:.2f}x</div>
                <div>at resource level {max_speedup_level}</div>
                <div class="{efficiency_class}">{efficiency:.1%} efficiency</div>
            </div>
"""

    # Close metrics container
    html_content += """        </div>
        
        <div class="card">
            <h3>Key Findings</h3>
            <ul>"""
    
    # Add key findings based on the analysis
    if scalability_analysis:
        if 'scalability_score' in scalability_analysis:
            score = scalability_analysis['scalability_score']
            html_content += f"\n                <li>Overall scalability score: <span class='highlight'>{score:.2f}/10</span></li>"
        
        if 'efficiency' in scalability_analysis:
            efficiency = scalability_analysis['efficiency']
            html_content += f"\n                <li>Resource utilization efficiency: <span class='highlight'>{efficiency:.2%}</span></li>"
        
        if 'best_model' in scalability_analysis:
            model = scalability_analysis['best_model']
            html_content += f"\n                <li>Best fitting scalability model: <span class='highlight'>{model}</span></li>"
            
        if 'bottleneck' in scalability_analysis:
            bottleneck = scalability_analysis['bottleneck']
            html_content += f"\n                <li>Potential bottleneck: <span class='highlight'>{bottleneck}</span></li>"
    
    html_content += """            
            </ul>
        </div>"""
    
    # Add test configuration section
    html_content += """
        <h2>Test Configuration</h2>
        <table>
            <tr>
                <th>Resource Level</th>
                <th>Throughput (req/s)</th>
                <th>Response Time (ms)</th>
                <th>Relative Speedup</th>
            </tr>"""
    
    # Add test configuration data
    for i, (level, throughput, response_time, speedup) in enumerate(zip(resource_levels, throughputs, response_times, speedups)):
        row_class = ""
        if i > 0 and 'error_percentages' in locals() and i < len(error_percentages):
            error_percentage = error_percentages[i]
            row_class = "" if error_percentage < 1 else "warning" if error_percentage < 5 else "critical"
        
        html_content += f"""            <tr class="{row_class}">
                <td>{level}</td>
                <td>{throughput:.2f}</td>
                <td>{response_time:.2f}</td>
                <td>{speedup:.2f}x</td>
            </tr>
"""
    
    html_content += """        </table>"""
    
    # Add Basic Scalability Metrics table
    html_content += """
        <h2>Basic Scalability Metrics</h2>
        <table>
            <thead>
                <tr>
                    <th>Resource Level</th>
                    <th>Throughput Speedup</th>
                    <th>Scalability Efficiency</th>
                </tr>
            </thead>
            <tbody>"""
    
    # Add scalability metrics data
    for i, (level, speedup, efficiency) in enumerate(zip(resource_levels, speedups, efficiencies)):
        # Add row class based on efficiency
        eff_class = "good" if efficiency > 0.85 else "warning" if efficiency > 0.6 else "critical"
        
        html_content += f"""                <tr>
                    <td>{level}</td>
                    <td>{speedup:.2f}x</td>
                    <td class="{eff_class}">{efficiency:.2%}</td>
                </tr>
"""
        
    html_content += """            </tbody>
        </table>"""
    
    # Add Visual Analysis section header
    html_content += """
        <h2>Visual Analysis</h2>
        <p>The following plots provide visual representation of the scalability characteristics.</p>"""
    
    # Add throughput plot if available
    if plot_paths and 'throughput' in plot_paths:
        html_content += f"""
        <div class="plot-container">
            <h3>Throughput vs. Resource Level</h3>
            <img src="{os.path.basename(plot_paths['throughput'])}" alt="Throughput Plot">
            <p class="plot-description">This plot shows how the system throughput changes as resources are added. 
            The trend line indicates the overall scaling pattern.</p>
        </div>
"""

    # Add response time plot if available
    if plot_paths and 'response_time' in plot_paths:
        html_content += f"""
        <div class="plot-container">
            <h3>Response Time vs. Resource Level</h3>
            <img src="{os.path.basename(plot_paths['response_time'])}" alt="Response Time Plot">
            <p class="plot-description">This plot illustrates how response times are affected by resource scaling. 
            Lower values indicate better performance.</p>
        </div>
"""

    # Add speedup plot if available
    if plot_paths and 'speedup' in plot_paths:
        html_content += f"""
        <div class="plot-container">
            <h3>Speedup vs. Resource Level</h3>
            <img src="{os.path.basename(plot_paths['speedup'])}" alt="Speedup Plot">
            <p class="plot-description">This plot compares actual speedup against ideal linear speedup. 
            The gap between actual and ideal lines indicates efficiency loss as resources scale.</p>
        </div>
"""
    
    # Add model comparison plot if available
    if plot_paths and 'models_comparison' in plot_paths:
        html_content += f"""
        <div class="plot-container">
            <h3>Scalability Models Comparison</h3>
            <img src="{os.path.basename(plot_paths['models_comparison'])}" alt="Models Comparison Plot">
            <p class="plot-description">This plot compares the actual speedup with predictions from different scalability laws. 
            The closest model to actual data points indicates which theoretical model best describes the system's scaling behavior.</p>
        </div>
"""

    # Add theoretical projections plot if available
    if plot_paths and 'theoretical_projections' in plot_paths:
        html_content += f"""
        <div class="plot-container">
            <h3>Theoretical Scalability Projections</h3>
            <img src="{os.path.basename(plot_paths['theoretical_projections'])}" alt="Theoretical Projections Plot">
            <p class="plot-description">This plot shows theoretical projections of different scalability models based on observed data. 
            It predicts how the system might scale with additional resources beyond those tested.</p>
        </div>
"""

    # Add model characteristics plot if available
    if plot_paths and 'model_characteristics' in plot_paths:
        html_content += f"""
        <div class="plot-container">
            <h3>Comparative Scalability Model Characteristics</h3>
            <img src="{os.path.basename(plot_paths['model_characteristics'])}" alt="Model Characteristics Plot">
            <p class="plot-description">This educational plot illustrates the fundamental differences between various scalability models 
            with different parameters. It helps identify which theoretical model best describes your system's behavior.</p>
        </div>
"""

    # Add efficiency plot if available
    if plot_paths and 'efficiency' in plot_paths:
        html_content += f"""
        <div class="plot-container">
            <h3>Scalability Efficiency Analysis</h3>
            <img src="{os.path.basename(plot_paths['efficiency'])}" alt="Scalability Efficiency Plot">
            <p class="plot-description">This plot shows how efficiently your system scales as resources increase. 
            The efficiency is calculated as speedup divided by resource ratio, with 100% representing perfect linear scaling. 
            Declining efficiency indicates diminishing returns from additional resources.</p>
        </div>
"""
    
    # Add efficiency heatmap if available
    if plot_paths and 'heatmap' in plot_paths:
        html_content += f"""
        <div class="plot-container">
            <h3>Efficiency Heatmap</h3>
            <img src="{os.path.basename(plot_paths['heatmap'])}" alt="Efficiency Heatmap">
            <p class="plot-description">This heatmap visualizes scaling efficiency between different resource levels. 
            Each cell shows the efficiency when scaling from the baseline (row) to the target (column) resource level. 
            Values close to 1.0 (green) indicate good scaling efficiency.</p>
        </div>
"""

    # Add cost efficiency plot if available
    if plot_paths and 'cost_efficiency' in plot_paths:
        html_content += f"""
        <div class="plot-container">
            <h3>Cost Efficiency Analysis</h3>
            <img src="{os.path.basename(plot_paths['cost_efficiency'])}" alt="Cost Efficiency Analysis">
            <p class="plot-description">This dual visualization shows the relationship between cost and performance (left) 
            and the performance-to-cost ratio (right) for different resource levels. 
            Higher values in the right plot indicate more cost-effective configurations.</p>
        </div>
"""
    
    # Add algorithm complexity analysis if available
    if plot_paths and 'algorithm_complexity_plot' in plot_paths and os.path.exists(plot_paths['algorithm_complexity_plot']):
        html_content += """
    <h2>Algorithm Complexity Analysis</h2>
    <div class="plot-container">
        <img src="%s" alt="Algorithm Complexity Plot">
        <p class="plot-description">
            This visualization analyzes the algorithmic complexity patterns in the system's scaling behavior. 
            It compares the observed performance data against different computational complexity classes 
            (O(1), O(log n), O(n), O(n log n), O(n²), etc.) to identify which mathematical model best 
            describes how the system's response time or computational cost scales with input size.
        </p>
        <p class="plot-description">
            <strong>Interpretation Guide:</strong>
            <ul>
                <li><strong>O(1) - Constant time:</strong> Response time remains constant regardless of input size. Ideal for scalable systems.</li>
                <li><strong>O(log n) - Logarithmic time:</strong> Response time grows very slowly as input size increases. Excellent scaling.</li>
                <li><strong>O(n) - Linear time:</strong> Response time grows proportionally with input size. Good scaling.</li>
                <li><strong>O(n log n) - Linearithmic time:</strong> Response time grows slightly faster than proportionally. Common in efficient algorithms.</li>
                <li><strong>O(n²) - Quadratic time:</strong> Response time grows with the square of input size. Indicates potential scaling issues.</li>
            </ul>
            The best-fitting curve indicates the dominant computational pattern in your system, which helps identify bottlenecks and optimization opportunities.
        </p>
    </div>
""" % os.path.basename(plot_paths['algorithm_complexity_plot'])

    # Add load scalability analysis if available
    if plot_paths and 'load_scalability_plot' in plot_paths and os.path.exists(plot_paths['load_scalability_plot']):
        html_content += """
    <h2>Load Scalability Analysis</h2>
    <div class="plot-container">
        <img src="%s" alt="Load Scalability Analysis">
        <p class="plot-description">
            This visualization analyzes how system performance metrics (throughput and response time) change with increasing load levels. 
            It helps identify the saturation point where adding more load no longer improves throughput and may degrade response time.
        </p>
        <p class="plot-description">
            <strong>Key Indicators:</strong>
            <ul>
                <li><strong>Linear Growth Region:</strong> The initial section where throughput increases linearly with load indicates efficient resource utilization.</li>
                <li><strong>Saturation Point:</strong> The load level where throughput begins to level off, indicating system resources becoming fully utilized.</li>
                <li><strong>Degradation Region:</strong> The area where throughput may decrease while response time continues to increase, indicating system overload.</li>
                <li><strong>Optimal Load Point:</strong> The load level that provides the best balance between high throughput and acceptable response time.</li>
            </ul>
            Understanding these patterns helps in capacity planning and identifying the optimal operating range for your system.
        </p>
    </div>
""" % os.path.basename(plot_paths['load_scalability_plot'])

    # Add load capacity model if available
    if plot_paths and 'capacity_model_plot' in plot_paths and os.path.exists(plot_paths['capacity_model_plot']):
        html_content += """
    <h2>Load Capacity Model</h2>
    <div class="plot-container">
        <img src="%s" alt="Load Capacity Model">
        <p class="plot-description">
            This visualization presents a predictive capacity model showing the theoretical maximum throughput capacity 
            of the system and the limiting factors that constrain it. The model applies the Universal Scalability Law 
            to predict system behavior under various load conditions and resource configurations.
        </p>
        <p class="plot-description">
            <strong>Model Components:</strong>
            <ul>
                <li><strong>Theoretical Maximum:</strong> The asymptotic maximum throughput the system can achieve under ideal conditions.</li>
                <li><strong>Contention Factor (α):</strong> Represents resource conflicts and serialization that limit scalability.</li>
                <li><strong>Coherency Factor (β):</strong> Represents coordination overhead that causes performance to eventually decrease with added load.</li>
                <li><strong>Optimal Operating Point:</strong> The load level that maximizes throughput before coherency costs cause degradation.</li>
            </ul>
            The capacity model helps predict how the system will behave beyond the tested load levels, enabling more accurate capacity planning and infrastructure sizing decisions.
        </p>
    </div>
""" % os.path.basename(plot_paths['capacity_model_plot'])

    # Add Optimization Suggestions section if available
    if scalability_analysis and 'optimization_suggestions' in scalability_analysis:
        suggestions = scalability_analysis.get('optimization_suggestions', [])
        if suggestions:
            html_content += """        <h2>Optimization Suggestions</h2>
"""
            
            for suggestion in suggestions:
                html_content += f"""        <div class="recommendation">
            {suggestion}
        </div>
"""

    # Close HTML tags
    html_content += """    </div>
</body>
</html>
"""

    # Save the HTML report
    report_path = os.path.join(output_dir, 'scalability_report.html')
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    # Copy plot images if they exist
    if plot_paths:
        for plot_type, plot_path in plot_paths.items():
            if os.path.exists(plot_path):
                plot_filename = os.path.basename(plot_path)
                dest_path = os.path.join(output_dir, plot_filename)
                if plot_path != dest_path:  # Don't copy if source and destination are the same
                    with open(plot_path, 'rb') as src_file:
                        with open(dest_path, 'wb') as dest_file:
                            dest_file.write(src_file.read())
                print(f"Added visualization: {plot_type} - {plot_filename}")
    
    print(f"Enhanced HTML report saved to {report_path}")
    return report_path


def convert_markdown_to_html(markdown_path, output_dir):
    """
    Convert a Markdown report to HTML with enhanced styling
    
    Args:
        markdown_path (str): Path to the Markdown report
        output_dir (str): Directory to save the HTML report
        
    Returns:
        str: Path to the generated HTML report
    """
    # Read the Markdown file
    with open(markdown_path, 'r') as f:
        markdown_content = f.read()
    
    # Convert Markdown to HTML
    html_content = markdown2.markdown(
        markdown_content,
        extras=['tables', 'fenced-code-blocks', 'header-ids']
    )
    
    # Create CSS styling
    css_style = """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        .container {
            width: 90%;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #fff;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #2c3e50;
            margin-top: 30px;
        }
        h1 {
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f9f9f9;
        }
        code {
            background: #f8f8f8;
            border-radius: 3px;
            padding: 2px 5px;
            font-family: 'Courier New', Courier, monospace;
        }
        img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        strong {
            color: #2980b9;
        }
    """
    
    # Add HTML wrapper with CSS styling
    styled_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scalability Analysis Report</title>
    <style>
{css_style}
    </style>
</head>
<body>
    <div class="container">
        {html_content}
    </div>
</body>
</html>"""
    
    # Save the HTML report
    html_path = os.path.join(output_dir, 'scalability_report_from_md.html')
    with open(html_path, 'w') as f:
        f.write(styled_html)
    
    print(f"Markdown report converted to HTML: {html_path}")
    return html_path
