#!/usr/bin/env python3
"""
Scalability Reporting Module - Markdown Generator

This module provides functionality for generating Markdown reports from
scalability analysis results.

Part of the modular scalability analysis framework
"""

import os
from datetime import datetime


def generate_markdown_report(analysis_results, output_dir):
    """
    Generate a Markdown report from scalability analysis results
    
    Args:
        analysis_results (list): List of dictionaries with analysis results
        output_dir (str): Directory to save the report
        
    Returns:
        str: Path to the generated report
    """
    # Sort results by resource level for consistent reporting
    sorted_results = sorted(analysis_results, key=lambda x: x.get('resource_level', 0))
    
    # Extract resource levels and metrics for comparison
    resource_levels = [r.get('resource_level', 0) for r in sorted_results]
    throughputs = [r.get('metrics', {}).get('throughput', 0) for r in sorted_results]
    response_times = [r.get('metrics', {}).get('avg_response_time', 0) for r in sorted_results]
    
    # Calculate speedups relative to the baseline
    baseline_throughput = throughputs[0] if throughputs else 1
    speedups = [t / baseline_throughput for t in throughputs]
    
    # Start building the Markdown content
    report_md = "# Scalability Analysis Report\n\n"
    report_md += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Executive Summary section
    report_md += "## Executive Summary\n\n"
    if len(resource_levels) > 1:
        report_md += f"This report analyzes the scalability characteristics of the system under test across {len(resource_levels)} "
        report_md += f"different resource levels (from {min(resource_levels)} to {max(resource_levels)}).\n\n"
    else:
        report_md += "This report analyzes the performance characteristics of the system under test.\n\n"
    
    report_md += "**Key Findings:**\n\n"
    if throughputs:
        max_throughput = max(throughputs)
        max_throughput_level = resource_levels[throughputs.index(max_throughput)]
        report_md += f"- **Maximum Throughput:** {max_throughput:.2f} requests/sec achieved at resource level {max_throughput_level}\n"
    
    if response_times:
        min_response_time = min(response_times)
        min_response_time_level = resource_levels[response_times.index(min_response_time)]
        report_md += f"- **Best Response Time:** {min_response_time:.2f} ms achieved at resource level {min_response_time_level}\n"
    
    if len(speedups) > 1:
        max_speedup = max(speedups[1:], default=1)  # Skip baseline
        max_speedup_level = resource_levels[speedups.index(max_speedup)]
        report_md += f"- **Maximum Speedup:** {max_speedup:.2f}x achieved at resource level {max_speedup_level} compared to baseline\n"
    
    # Add scalability model insights if available
    scalability_analysis = sorted_results[0].get('scalability_analysis', {})
    if scalability_analysis and 'interpretations' in scalability_analysis:
        interpretations = scalability_analysis.get('interpretations', {})
        if 'amdahl' in interpretations:
            assessment = interpretations['amdahl'].get('assessment', '')
            if assessment:
                report_md += f"- **Amdahl's Law Analysis:** {assessment}\n"
        
        if 'optimization_suggestions' in scalability_analysis and scalability_analysis['optimization_suggestions']:
            report_md += "\n**Optimization Suggestions:**\n\n"
            for suggestion in scalability_analysis['optimization_suggestions'][:2]:  # Limit to top 2 suggestions
                report_md += f"- {suggestion}\n"
    
    # Detailed Performance Metrics section
    report_md += "\n## Detailed Performance Metrics\n\n"
    report_md += "| Resource Level | Throughput (req/s) | Avg Response Time (ms) | Error % |\n"
    report_md += "|---|---|---|---|\n"
    
    for i, result in enumerate(sorted_results):
        metrics = result.get('metrics', {})
        level = result.get('resource_level', 0)
        throughput = metrics.get('throughput', 0)
        response_time = metrics.get('avg_response_time', 0)
        error_percentage = metrics.get('error_percentage', 0)
        
        report_md += f"| {level} | {throughput:.2f} | {response_time:.2f} | {error_percentage:.2f} |\n"
    
    # Basic Scalability Metrics section
    if len(resource_levels) > 1:
        report_md += "\n## Basic Scalability Metrics\n\n"
        report_md += "| Resource Level | Throughput Speedup | Scalability Efficiency |\n"
        report_md += "|---|---|---|\n"
        
        for i, level in enumerate(resource_levels):
            speedup = speedups[i]
            # Calculate efficiency as speedup / resource_level relative to baseline
            relative_resources = level / resource_levels[0]
            efficiency = speedup / relative_resources if relative_resources > 0 else 0
            
            report_md += f"| {level} | {speedup:.2f}x | {efficiency:.2%} |\n"
    
    # Advanced Scalability Analysis section if available
    if scalability_analysis and 'models' in scalability_analysis:
        report_md += "\n## Advanced Scalability Analysis\n\n"
        models = scalability_analysis.get('models', {})
        
        if 'amdahl' in models:
            p = models['amdahl']
            report_md += f"### Amdahl's Law Analysis\n\n"
            report_md += f"- Parallelizable portion: **{p:.2%}**\n"
            report_md += f"- Serial portion: **{1-p:.2%}**\n"
            report_md += f"- Theoretical maximum speedup: **{1/(1-p):.2f}x**\n"
        
        if 'gustafson' in models:
            p = models['gustafson']
            report_md += f"\n### Gustafson's Law Analysis\n\n"
            report_md += f"- Scalable portion: **{p:.2%}**\n"
            report_md += f"- Fixed portion: **{1-p:.2%}**\n"
        
        if 'usl' in models:
            sigma, kappa = models['usl']
            report_md += f"\n### Universal Scalability Law Analysis\n\n"
            report_md += f"- Contention factor (σ): **{sigma:.4f}**\n"
            report_md += f"- Coherency factor (κ): **{kappa:.4f}**\n"
            
            # Calculate peak concurrency point
            if kappa > 0:
                peak_n = (1 - sigma) / (2 * kappa) if (1 - sigma) > 0 else 1
                report_md += f"- Optimal concurrency: **{peak_n:.2f}** resources\n"
        
        # Add interpretations if available
        if 'interpretations' in scalability_analysis:
            interpretations = scalability_analysis.get('interpretations', {})
            report_md += "\n### Model Interpretations\n\n"
            
            for model_name, interp in interpretations.items():
                if 'assessment' in interp:
                    report_md += f"- **{model_name.capitalize()}:** {interp['assessment']}\n"
    
    # Visual Analysis section
    report_md += "\n## Visual Analysis\n\n"
    report_md += "Refer to the generated plots for visual representation of the scalability characteristics.\n"
    report_md += "Key visualizations include throughput vs. resource level, response time trends, and comparative analysis of " 
    report_md += "actual speedup against theoretical models.\n\n"
    report_md += "The plots provide insights into:\n"
    report_md += "- System scaling behavior across different resource levels\n"
    report_md += "- Deviation from ideal linear scaling\n"
    report_md += "- Efficiency trends and potential bottlenecks\n"
    
    # Add advanced visualization descriptions if relevant plots exist
    if len(resource_levels) >= 2:
        report_md += "\n### Advanced Visualization Analysis\n\n"
        
        # Efficiency analysis
        report_md += "#### Scalability Efficiency Analysis\n\n"
        report_md += "The efficiency plot shows how effectively your system utilizes additional resources. "
        report_md += "Efficiency is calculated as speedup divided by the resource ratio, with 100% indicating perfect linear scaling.\n\n"
        
        # Calculate and add efficiency metrics
        baseline_throughput = throughputs[0]
        baseline_resource = resource_levels[0]
        speedups = [t / baseline_throughput for t in throughputs]
        resource_ratios = [r / baseline_resource for r in resource_levels]
        efficiencies = [s / r for s, r in zip(speedups, resource_ratios)]
        
        last_efficiency = efficiencies[-1] if efficiencies else 0
        report_md += f"**Current Scaling Efficiency:** {last_efficiency:.1%} at {resource_levels[-1]} resources\n\n"
        
        if last_efficiency > 0.9:
            report_md += "**Observation:** Excellent scaling efficiency indicates your system has minimal serialization bottlenecks\n\n"
        elif last_efficiency > 0.7:
            report_md += "**Observation:** Good scaling efficiency indicates your system parallelizes well with some overhead\n\n"
        else:
            report_md += "**Observation:** Lower scaling efficiency suggests significant serialization or contention in your system\n\n"
        
        # Cost efficiency analysis
        report_md += "#### Cost Efficiency Analysis\n\n"
        report_md += "The cost efficiency visualization shows the relationship between cost and performance. "
        report_md += "It identifies which configuration provides the best throughput per cost unit.\n\n"
        
        # Calculate cost efficiency metrics
        cost_model = lambda x: x  # Default linear cost model
        costs = [cost_model(r) for r in resource_levels]
        throughput_per_cost = [t / c if c > 0 else 0 for t, c in zip(throughputs, costs)]
        most_efficient_idx = throughput_per_cost.index(max(throughput_per_cost)) if throughput_per_cost else 0
        
        report_md += f"**Most Cost-Efficient Configuration:** {resource_levels[most_efficient_idx]} resources\n\n"
        
        if most_efficient_idx == len(resource_levels) - 1:
            report_md += "**Recommendation:** Consider testing with even more resources as cost efficiency is still increasing\n"
        elif most_efficient_idx == 0:
            report_md += "**Recommendation:** The smallest configuration is most cost-efficient; consider if performance meets your needs\n"
        else:
            report_md += f"**Recommendation:** {resource_levels[most_efficient_idx]} resources provides the best balance of performance and cost\n"
    
    # Add information about theoretical projections if available
    if scalability_analysis and scalability_analysis.get('insufficient_data', False):
        theoretical_projections = scalability_analysis.get('theoretical_projections', {})
        if theoretical_projections:
            report_md += "\n## Theoretical Projections Based on Limited Data\n\n"
            report_md += "Since only a limited number of resource levels were tested, theoretical projections "
            report_md += "have been generated based on the available data points. These projections should be considered "
            report_md += "estimates that need validation with additional measurements.\n\n"
            
            if 'amdahl' in theoretical_projections:
                p = theoretical_projections['amdahl'].get('parallelizable_fraction', 0)
                max_speedup = theoretical_projections['amdahl'].get('estimated_max_speedup', 0)
                efficiency = theoretical_projections['amdahl'].get('observed_efficiency', 0)
                
                report_md += "### Amdahl's Law Projection\n\n"
                report_md += f"Based on the observed speedup efficiency of {efficiency:.2%}, Amdahl's Law suggests that "
                report_md += f"approximately **{p*100:.1f}%** of your workload is parallelizable.\n\n"
                report_md += f"**Implications:**\n\n"
                report_md += f"- Theoretical maximum speedup limit: **{max_speedup:.1f}x**\n"
                report_md += f"- As you add more resources beyond those tested, speedup will approach but never exceed this limit\n"
            
            if 'gustafson' in theoretical_projections:
                p = theoretical_projections['gustafson'].get('scalable_fraction', 0)
                
                report_md += "\n### Gustafson's Law Projection\n\n"
                report_md += f"Gustafson's Law suggests that **{p*100:.1f}%** of your workload scales with additional resources.\n\n"
                report_md += f"**Implications:**\n\n"
                report_md += f"- If workload size increases with resource count, your system may achieve better scaling than Amdahl's Law predicts\n"
    
    # Add advanced analysis sections if available
    first_result = sorted_results[0] if sorted_results else None
    if first_result and 'advanced_analysis' in first_result:
        advanced_analysis = first_result['advanced_analysis']
        
        # Add algorithm complexity analysis if available
        if 'algorithm_complexity' in advanced_analysis:
            complexity = advanced_analysis['algorithm_complexity']
            report_md += "\n## Algorithm Complexity Analysis\n\n"
            
            if 'interpretation' in complexity and complexity['interpretation'].get('success', False):
                interpretation = complexity['interpretation']
                best_model = interpretation.get('best_model', 'Unknown')
                confidence = interpretation.get('confidence', '')
                
                report_md += f"**Best fitting model:** {best_model}\n\n"
                report_md += f"**Confidence:** {confidence}\n\n"
                report_md += f"**Explanation:** {interpretation.get('explanation', '')}\n\n"
                report_md += f"**Implications:** {interpretation.get('implications', '')}\n\n"
                report_md += f"**Recommendations:** {interpretation.get('recommendations', '')}\n\n"
            else:
                report_md += "Insufficient data for reliable algorithm complexity analysis.\n\n"
        
        # Add load scalability analysis if available
        if 'load_scalability' in advanced_analysis:
            load = advanced_analysis['load_scalability']
            report_md += "\n## Load Scalability Analysis\n\n"
            
            if 'interpretation' in load and load['interpretation'].get('success', False):
                interpretation = load['interpretation']
                key_metrics = interpretation.get('key_metrics', {})
                
                report_md += f"**Saturation point:** {key_metrics.get('saturation_load', 'Unknown')} users/requests\n\n"
                report_md += f"**Optimal load point:** {key_metrics.get('optimal_load', 'Unknown')} users/requests\n\n"
                report_md += f"**Performance degradation observed:** {'Yes' if key_metrics.get('has_degradation', False) else 'No'}\n\n"
                
                report_md += "### Key Insights\n\n"
                for insight in interpretation.get('insights', []):
                    report_md += f"- {insight}\n"
                
                report_md += "\n### Recommendations\n\n"
                for recommendation in interpretation.get('recommendations', []):
                    report_md += f"- {recommendation}\n"
            else:
                report_md += "Insufficient data for reliable load scalability analysis.\n\n"
    
    # Optimization Suggestions section
    if scalability_analysis and 'optimization_suggestions' in scalability_analysis:
        report_md += "\n## Optimization Suggestions\n\n"
        
        if scalability_analysis['optimization_suggestions']:
            for suggestion in scalability_analysis['optimization_suggestions']:
                report_md += f"- {suggestion}\n"
        else:
            report_md += "No specific optimization suggestions based on current data.\n"
    
    # Save the Markdown report
    report_path = os.path.join(output_dir, 'scalability_report.md')
    with open(report_path, 'w') as f:
        f.write(report_md)
    
    print(f"Markdown report saved to {report_path}")
    return report_path


# Main entry point for direct script execution
if __name__ == "__main__":
    import argparse
    from scalability_core import analyze_jtl, create_output_dir
    
    parser = argparse.ArgumentParser(description='Scalability Reporting Module - Markdown Generator')
    parser.add_argument('--files', nargs='+', required=True, help='JTL files to analyze')
    parser.add_argument('--levels', nargs='+', type=int, required=True, 
                        help='Resource levels corresponding to files')
    parser.add_argument('--output-dir', type=str, help='Output directory for report')
    
    args = parser.parse_args()
    
    if len(args.files) != len(args.levels):
        print("Error: Number of files must match number of resource levels")
        exit(1)
    
    # Create output directory
    output_dir = args.output_dir or create_output_dir()
    
    # Analyze files
    analysis_results = []
    
    for file_path, level in zip(args.files, args.levels):
        metrics = analyze_jtl(file_path)
        if metrics:
            analysis_results.append({
                'file': file_path,
                'resource_level': level,
                'metrics': metrics
            })
    
    # Generate report
    if analysis_results:
        report_path = generate_markdown_report(analysis_results, output_dir)
        print(f"Report generated: {report_path}")
    else:
        print("Error: No valid data to generate report")
