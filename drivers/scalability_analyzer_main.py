#!/usr/bin/env python3
"""
Scalability Analyzer - Main Entry Point

This module serves as the main entry point for the modular scalability analysis framework.
It integrates all the components:
- Core JTL parsing and metrics calculation
- Scalability models and analysis
- Visualization generation
- Report generation in multiple formats

Usage:
    python scalability_analyzer_main.py --files jtl1.jtl jtl2.jtl --levels 2 4 --output-dir results

Author: JJ Shen
"""

import os
import sys
import argparse
from datetime import datetime

# Import component modules
from scalability_core import analyze_jtl, create_output_dir, save_results_json
from scalability_models import (
    amdahls_law, fit_amdahls_law,
    gustafsons_law, fit_gustafsons_law,
    universal_scalability_law, fit_universal_scalability_law,
    plot_scalability_models, plot_theoretical_projections,
    interpret_scalability_results, suggest_optimizations
)

# Import intrinsic and load scalability analysis modules
from algorithm_complexity import (
    analyze_algorithm_complexity,
    plot_algorithm_complexity,
    interpret_algorithm_complexity
)
from load_scalability import (
    analyze_load_scalability,
    plot_load_scalability,
    interpret_load_scalability
)

# Import visualization module
from scalability_visualization_basic import (
    plot_throughput_vs_resource,
    plot_response_time_vs_resource,
    plot_speedup_vs_resource,
    plot_comparative_throughput,
    plot_response_time_distribution,
    plot_efficiency_heatmap,
    plot_scalability_efficiency,
    plot_cost_efficiency,
    create_advanced_scalability_plots
)

# Import reporting modules
from scalability_reporting_md import generate_markdown_report
# Use the HTML reporting module
from scalability_reporting_html import generate_html_report
from scalability_reporting_docx import generate_docx_report


def perform_scalability_analysis(results):
    """
    Perform in-depth scalability analysis using various scalability models.
    
    Args:
        results (list): List of test results with metrics and resource levels
        
    Returns:
        dict: Analysis results including model parameters and interpretations
    """
    # Sort results by resource level for consistent analysis
    sorted_results = sorted(results, key=lambda x: x['resource_level'])
    
    # Extract resource levels and performance metrics
    resource_levels = [r['resource_level'] for r in sorted_results]
    throughputs = [r['metrics']['throughput'] for r in sorted_results]
    
    # Calculate speedups relative to the baseline
    baseline_throughput = throughputs[0]
    speedups = [t / baseline_throughput for t in throughputs]
    
    # Initialize results dictionary
    analysis = {
        'resource_levels': resource_levels,
        'actual_speedups': speedups,
        'models': {},
        'interpretations': {},
        'optimization_suggestions': [],
        'theoretical_projections': {},
        'insufficient_data': False
    }
    
    # Check if we have enough data points (at least 3 for reliable fitting)
    if len(resource_levels) >= 3:
        # Fit Amdahl's Law
        try:
            p_amdahl, error_amdahl = fit_amdahls_law(resource_levels, speedups)
            analysis['models']['amdahl'] = p_amdahl
            analysis['model_errors'] = {'amdahl': error_amdahl}
        except Exception as e:
            print(f"Warning: Could not fit Amdahl's Law: {e}")
        
        # Fit Gustafson's Law
        try:
            p_gustafson, error_gustafson = fit_gustafsons_law(resource_levels, speedups)
            analysis['models']['gustafson'] = p_gustafson
            analysis['model_errors']['gustafson'] = error_gustafson
        except Exception as e:
            print(f"Warning: Could not fit Gustafson's Law: {e}")
        
        # Fit Universal Scalability Law
        try:
            sigma, kappa, error_usl = fit_universal_scalability_law(resource_levels, speedups)
            analysis['models']['usl'] = (sigma, kappa)
            analysis['model_errors']['usl'] = error_usl
        except Exception as e:
            print(f"Warning: Could not fit Universal Scalability Law: {e}")
        
        # Generate interpretations of model parameters
        if analysis['models']:
            analysis['interpretations'] = interpret_scalability_results(analysis['models'])
            analysis['optimization_suggestions'] = suggest_optimizations(
                analysis['models'], analysis['interpretations']
            )
    else:
        # Not enough data points for reliable model fitting
        print("Warning: Not enough data points to fit scalability models. At least 3 resource levels are required.")
        analysis['insufficient_data'] = True
        
        # For limited data, we can still provide theoretical projections
        if len(resource_levels) == 2:
            # Calculate efficiency based on the two data points we have
            efficiency = speedups[1] / (resource_levels[1] / resource_levels[0])
            
            # Estimate Amdahl's Law parameter based on efficiency
            # Function to minimize for estimating p in Amdahl's Law
            def amdahl_error(p):
                predicted = amdahls_law(p, resource_levels[1] / resource_levels[0])
                return (predicted - speedups[1]) ** 2
            
            # Estimate p using optimization
            from scipy.optimize import minimize_scalar
            result = minimize_scalar(amdahl_error, bounds=(0, 1), method='bounded')
            p_amdahl = result.x
            
            # Theoretical maximum speedup according to Amdahl's Law
            max_speedup = 1 / (1 - p_amdahl) if p_amdahl < 1 else float('inf')
            
            # Store the theoretical projection for Amdahl's Law
            analysis['theoretical_projections']['amdahl'] = {
                'parallelizable_fraction': p_amdahl,
                'estimated_max_speedup': max_speedup,
                'observed_efficiency': efficiency
            }
            
            # Estimate Gustafson's Law parameter based on the same data
            p_gustafson = p_amdahl  # In simple cases, these are often similar
            
            # Store the theoretical projection for Gustafson's Law
            analysis['theoretical_projections']['gustafson'] = {
                'scalable_fraction': p_gustafson
            }
            
            # For USL, we can't reliably estimate parameters with just 2 points,
            # but we can provide typical values for illustration
            analysis['theoretical_projections']['usl'] = {
                'typical_sigma': 0.1,  # Typical contention factor
                'typical_kappa': 0.01  # Typical coherency delay factor
            }
    
    return analysis


def main():
    """Main entry point for the scalability analyzer"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Comprehensive Scalability Analyzer')
    parser.add_argument('--files', nargs='+', required=True, help='List of JTL file paths')
    parser.add_argument('--levels', nargs='+', type=int, required=True, help='Resource levels for each file')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for results')
    parser.add_argument('--formats', nargs='+', choices=['md', 'html', 'docx'], default=['md', 'html', 'docx'],
                       help='Report formats to generate')
    parser.add_argument('--sla', type=float, default=None, help='SLA threshold for response time (ms)')
    parser.add_argument('--show-plots', action='store_true', help='Display plots during analysis')
    parser.add_argument('--comparative', action='store_true', 
                       help='Generate comparative analysis plots between different configurations')
    parser.add_argument('--advanced', action='store_true',
                       help='Generate advanced visualization plots including efficiency analysis')
    parser.add_argument('--cost-model', choices=['linear', 'quadratic', 'custom'],
                       default='linear', help='Cost model for efficiency analysis')
    parser.add_argument('--configs', nargs='+', help='Configuration names for each JTL file')
    
    # New arguments for intrinsic and load scalability analysis
    parser.add_argument('--algorithm-complexity', action='store_true',
                        help='Perform algorithm complexity analysis to identify computational complexity')
    parser.add_argument('--load-scalability', action='store_true',
                        help='Perform load scalability analysis to identify saturation points')
    parser.add_argument('--load-levels', nargs='+', type=float,
                        help='Load levels for load scalability analysis (e.g., concurrent users)')
    parser.add_argument('--execution-times', nargs='+', type=float,
                        help='Execution times for algorithm complexity analysis')
    
    args = parser.parse_args()
    
    # Validate arguments
    if len(args.files) != len(args.levels):
        print("Error: Number of files must match number of resource levels")
        sys.exit(1)
    
    if args.configs and len(args.configs) != len(args.files):
        print("Error: Number of configurations must match number of files")
        sys.exit(1)
    
    # Create descriptive output directory with meaningful name
    if not args.output_dir:
        # Create descriptive name based on test configuration
        test_description = "_".join(args.configs) if args.configs else None
        if not test_description:
            # Use resource levels for directory name if configs not provided
            test_description = f"Nodes{'-'.join(str(level) for level in args.levels)}"
        
        # Include analysis type in the name
        name_parts = [test_description]
        if args.advanced:
            name_parts.append("Advanced")
        if args.comparative:
            name_parts.append("Comparative")
            
        # Create the directory
        output_dir = create_output_dir(
            custom_name=f"ScalabilityAnalysis_{'_'.join(name_parts)}",
            resource_levels=args.levels
        )
    else:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Analysis results will be saved to: {output_dir}")
    
    # Process JTL files
    analysis_results = []
    dataframes = []
    
    print(f"Processing {len(args.files)} JTL files...")
    for i, (file_path, level) in enumerate(zip(args.files, args.levels)):
        print(f"Analyzing file {i+1}/{len(args.files)}: {os.path.basename(file_path)}")
        metrics = analyze_jtl(file_path)
        
        if metrics:
            result = {
                'file': file_path,
                'resource_level': level,
                'metrics': metrics,
                'config': args.configs[i] if args.configs and i < len(args.configs) else f"Config {level}"
            }
            analysis_results.append(result)
    
    if not analysis_results:
        print("Error: No valid data could be extracted from the JTL files")
        sys.exit(1)
    
    print(f"Successfully processed {len(analysis_results)} out of {len(args.files)} JTL files")
    
    # Sort results by resource level
    sorted_results = sorted(analysis_results, key=lambda x: x['resource_level'])
    
    # Extract key metrics for visualization
    resource_levels = [r['resource_level'] for r in sorted_results]
    throughputs = [r['metrics']['throughput'] for r in sorted_results]
    response_times = [r['metrics']['avg_response_time'] for r in sorted_results]
    
    # Generate basic plots
    print("Generating basic performance plots...")
    plot_paths = {}
    
    # Throughput vs Resource Level plot
    throughput_plot = plot_throughput_vs_resource(
        resource_levels, throughputs, output_dir, args.show_plots
    )
    plot_paths['throughput'] = throughput_plot
    
    # Response Time vs Resource Level plot
    response_time_plot = plot_response_time_vs_resource(
        resource_levels, response_times, output_dir, args.sla, args.show_plots
    )
    plot_paths['response_time'] = response_time_plot
    
    # Calculate speedups
    baseline_throughput = throughputs[0] if throughputs else 1
    speedups = [t / baseline_throughput for t in throughputs]
    
    # Speedup vs Resource Level plot
    speedup_plot = plot_speedup_vs_resource(
        resource_levels, speedups, output_dir, args.show_plots
    )
    plot_paths['speedup'] = speedup_plot
    
    # Perform scalability analysis
    print("Performing scalability analysis...")
    scalability_analysis = perform_scalability_analysis(sorted_results)
    
    # Add scalability analysis to the first result for reporting
    for result in sorted_results:
        result['scalability_analysis'] = scalability_analysis
    
    # Generate model plots if we have model parameters
    if scalability_analysis['models']:
        print("Generating scalability model plots...")
        model_plots = plot_scalability_models(
            resource_levels,
            speedups,
            output_dir,
            scalability_analysis['models'],
            args.show_plots
        )
        plot_paths.update(model_plots)
        
    # Generate advanced plots if requested
    if args.advanced:
        print("Generating advanced visualization plots...")
        
        # Define cost model based on user selection
        cost_model = None
        if args.cost_model == 'linear':
            cost_model = lambda x: x  # Cost is directly proportional to resources
        elif args.cost_model == 'quadratic':
            cost_model = lambda x: x**2  # Cost grows quadratically with resources
        
        # Generate advanced scalability plots
        advanced_plots = create_advanced_scalability_plots(
            resource_levels,
            throughputs,
            output_dir,
            scalability_analysis.get('models', {}),
            args.show_plots
        )
        plot_paths.update(advanced_plots)
        
        # Generate cost efficiency plot with the specified cost model
        if cost_model:
            cost_plot = plot_cost_efficiency(
                resource_levels,
                throughputs,
                output_dir,
                cost_model,
                args.show_plots
            )
            plot_paths['cost_model'] = cost_plot
    
    # Generate theoretical projection plots if we have limited data
    if scalability_analysis['insufficient_data'] and 'theoretical_projections' in scalability_analysis:
        print("Generating theoretical projection plots...")
        projection_plots = plot_theoretical_projections(
            resource_levels,
            speedups,
            output_dir,
            scalability_analysis['theoretical_projections'],
            max_projection=16,  # Project up to 16x resources
            show_plots=args.show_plots
        )
        plot_paths.update(projection_plots)
    
    # Always generate comparative model plots for educational purposes
    print("Generating comparative model characteristics plot...")
    comparative_plots = plot_theoretical_projections(
        resource_levels,
        speedups,
        output_dir,
        show_plots=args.show_plots
    )
    plot_paths.update(comparative_plots)
    
    # Perform algorithm complexity analysis if requested
    if args.algorithm_complexity:
        print("Performing algorithm complexity analysis...")
        # Use resource levels as input sizes if not specified
        input_sizes = args.load_levels if args.load_levels else resource_levels
        # Use execution times if specified, otherwise use response times
        exec_times = args.execution_times if args.execution_times else response_times
        
        if len(input_sizes) != len(exec_times):
            print("Warning: Number of input sizes must match execution times for algorithm complexity analysis")
        else:
            # Run the analysis
            complexity_analysis = analyze_algorithm_complexity(input_sizes, exec_times)
            
            if complexity_analysis.get("success", False):
                # Generate visualization
                complexity_plot = plot_algorithm_complexity(
                    input_sizes, exec_times, complexity_analysis, output_dir, args.show_plots
                )
                plot_paths['algorithm_complexity'] = complexity_plot
                
                # Get interpretation
                complexity_interpretation = interpret_algorithm_complexity(complexity_analysis)
                
                # Add results to the analysis for reporting
                for result in sorted_results:
                    if 'advanced_analysis' not in result:
                        result['advanced_analysis'] = {}
                    result['advanced_analysis']['algorithm_complexity'] = {
                        'analysis': complexity_analysis,
                        'interpretation': complexity_interpretation
                    }
                
                print(f"Algorithm complexity analysis complete. Best fit: {complexity_analysis['best_fit']}")
            else:
                print(f"Algorithm complexity analysis failed: {complexity_analysis.get('error', 'Unknown error')}")
    
    # Perform load scalability analysis if requested
    if args.load_scalability:
        print("Performing load scalability analysis...")
        # Use resource levels as load levels if not specified
        load_levels = args.load_levels if args.load_levels else resource_levels
        
        if len(load_levels) < 3:
            print("Warning: Load scalability analysis requires at least 3 data points")
        else:
            # Run the analysis
            load_analysis = analyze_load_scalability(load_levels, throughputs, response_times)
            
            if load_analysis.get("success", False):
                # Generate visualization
                load_plots = plot_load_scalability(
                    load_levels, throughputs, response_times, load_analysis, output_dir, args.show_plots
                )
                plot_paths.update(load_plots)
                
                # Get interpretation
                load_interpretation = interpret_load_scalability(load_analysis)
                
                # Add results to the analysis for reporting
                for result in sorted_results:
                    if 'advanced_analysis' not in result:
                        result['advanced_analysis'] = {}
                    result['advanced_analysis']['load_scalability'] = {
                        'analysis': load_analysis,
                        'interpretation': load_interpretation
                    }
                
                print(f"Load scalability analysis complete. Saturation point: {load_analysis['saturation_point']['load']} users")
            else:
                print(f"Load scalability analysis failed: {load_analysis.get('error', 'Unknown error')}")

    
    # Generate reports in requested formats
    print("Generating reports...")
    report_paths = {}
    
    if 'md' in args.formats:
        print("Generating Markdown report...")
        md_path = generate_markdown_report(sorted_results, output_dir)
        report_paths['markdown'] = md_path
    
    if 'html' in args.formats:
        print("Generating HTML report...")
        html_path = generate_html_report(sorted_results, output_dir, plot_paths)
        report_paths['html'] = html_path
    
    if 'docx' in args.formats:
        print("Generating DOCX report...")
        docx_path = generate_docx_report(sorted_results, output_dir, plot_paths)
        report_paths['docx'] = docx_path
    
    # Save complete results as JSON for future reference
    save_results_json({
        'analysis_results': sorted_results,
        'scalability_analysis': scalability_analysis,
        'plot_paths': plot_paths,
        'report_paths': report_paths
    }, output_dir)
    
    print("\nScalability analysis completed successfully!")
    print(f"Results saved to: {output_dir}")
    for fmt, path in report_paths.items():
        print(f"- {fmt.upper()} report: {os.path.basename(path)}")
    print("\nPlots generated:")
    for plot_type, path in plot_paths.items():
        print(f"- {plot_type}: {os.path.basename(path)}")


if __name__ == "__main__":
    main()
