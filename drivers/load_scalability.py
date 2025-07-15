#!/usr/bin/env python3
"""
Load Scalability Analysis Module

This module provides functions to analyze how a system's performance changes
under different load levels while resources remain constant. It helps identify
saturation points, capacity limits, and performance degradation patterns.

Author: JJ Shen
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import optimize


def analyze_load_scalability(load_levels, throughputs, response_times):
    """
    Analyze how performance metrics change with increasing load
    
    Args:
        load_levels (list): Load levels (e.g., concurrent users, request rates)
        throughputs (list): Corresponding throughput values
        response_times (list): Corresponding response time values
        
    Returns:
        dict: Analysis results including saturation points and capacity limits
    """
    if len(load_levels) < 3:
        return {
            "success": False,
            "error": "Insufficient data points. At least 3 points needed for reliable analysis."
        }
    
    # Convert to numpy arrays
    x = np.array(load_levels)
    throughput_y = np.array(throughputs)
    response_time_y = np.array(response_times)
    
    # Identify throughput saturation point using the elbow method
    try:
        # Calculate the rate of change in throughput
        throughput_gradient = np.gradient(throughput_y, x)
        
        # Find where gradient drops significantly (less than half of max gradient)
        max_gradient = np.max(throughput_gradient)
        gradient_threshold = max_gradient * 0.5
        saturation_indices = np.where(throughput_gradient < gradient_threshold)[0]
        
        if len(saturation_indices) > 0:
            saturation_index = saturation_indices[0]
            saturation_load = x[saturation_index]
            saturation_throughput = throughput_y[saturation_index]
        else:
            # No saturation detected within the measured range
            saturation_index = len(x) - 1
            saturation_load = x[-1]
            saturation_throughput = throughput_y[-1]
            
        # Check if throughput decreases after reaching a peak
        peak_index = np.argmax(throughput_y)
        peak_load = x[peak_index]
        peak_throughput = throughput_y[peak_index]
        
        has_degradation = peak_index < len(x) - 1 and peak_throughput > throughput_y[-1]
    except Exception as e:
        return {
            "success": False,
            "error": f"Error analyzing throughput saturation: {str(e)}"
        }
    
    # Analyze response time inflection point (where it starts increasing rapidly)
    try:
        # Calculate the rate of change in response time
        response_time_gradient = np.gradient(response_time_y, x)
        
        # Find where gradient increases significantly (more than twice the initial gradient)
        initial_gradient = response_time_gradient[0] if len(response_time_gradient) > 0 else 0
        inflection_threshold = initial_gradient * 2.0 if initial_gradient > 0 else 0.001
        inflection_indices = np.where(response_time_gradient > inflection_threshold)[0]
        
        if len(inflection_indices) > 0:
            inflection_index = inflection_indices[0]
            inflection_load = x[inflection_index]
            inflection_response_time = response_time_y[inflection_index]
        else:
            # No clear inflection point
            inflection_index = None
            inflection_load = None
            inflection_response_time = None
    except Exception as e:
        return {
            "success": False,
            "error": f"Error analyzing response time inflection: {str(e)}"
        }
    
    # Fit Little's Law relationship (Throughput = Load / Response Time)
    try:
        # Calculate theoretical throughput based on Little's Law
        littles_law_throughput = x / response_time_y
        
        # Calculate correlation between actual and theoretical throughput
        correlation = np.corrcoef(throughput_y, littles_law_throughput)[0, 1]
    except Exception:
        correlation = None
    
    # Determine optimal load level for best performance
    try:
        # Define a simple scoring function balancing throughput and response time
        # Normalize both metrics to 0-1 range
        norm_throughput = throughput_y / np.max(throughput_y) if np.max(throughput_y) > 0 else 0
        norm_response_time = 1 - (response_time_y / np.max(response_time_y)) if np.max(response_time_y) > 0 else 0
        
        # Score is weighted sum with more weight on throughput
        scores = 0.7 * norm_throughput + 0.3 * norm_response_time
        optimal_index = np.argmax(scores)
        optimal_load = x[optimal_index]
        optimal_throughput = throughput_y[optimal_index]
        optimal_response_time = response_time_y[optimal_index]
    except Exception:
        optimal_load = None
        optimal_throughput = None
        optimal_response_time = None
    
    # Return results
    return {
        "success": True,
        "saturation_point": {
            "load": saturation_load,
            "throughput": saturation_throughput,
            "index": saturation_index
        },
        "peak_performance": {
            "load": peak_load,
            "throughput": peak_throughput,
            "index": peak_index
        },
        "response_time_inflection": {
            "load": inflection_load,
            "response_time": inflection_response_time,
            "index": inflection_index
        },
        "optimal_load_point": {
            "load": optimal_load,
            "throughput": optimal_throughput,
            "response_time": optimal_response_time
        },
        "has_degradation": has_degradation,
        "littles_law_correlation": correlation
    }


def plot_load_scalability(load_levels, throughputs, response_times, analysis_results, output_dir, show_plot=False):
    """
    Generate plots showing load scalability analysis
    
    Args:
        load_levels (list): Load levels (e.g., concurrent users, request rates)
        throughputs (list): Corresponding throughput values
        response_times (list): Corresponding response time values
        analysis_results (dict): Results from analyze_load_scalability
        output_dir (str): Directory to save the plots
        show_plot (bool): Whether to display the plots
        
    Returns:
        dict: Paths to saved plots
    """
    if not analysis_results.get("success", False):
        print("Error: Cannot generate load scalability plots due to analysis failure")
        return {}
    
    plot_paths = {}
    
    # Create throughput vs load plot
    plt.figure(figsize=(12, 6))
    
    # Create plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    # Plot throughput on primary y-axis
    ax1.plot(load_levels, throughputs, 'bo-', linewidth=2, markersize=8, label='Throughput')
    ax1.set_xlabel('Load Level (Users/Requests)', fontsize=12)
    ax1.set_ylabel('Throughput (req/s)', fontsize=12, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Plot response time on secondary y-axis
    ax2.plot(load_levels, response_times, 'ro-', linewidth=2, markersize=8, label='Response Time')
    ax2.set_ylabel('Response Time (ms)', fontsize=12, color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Highlight important points if they exist
    saturation_point = analysis_results.get("saturation_point", {})
    if saturation_point.get("index") is not None:
        saturation_index = saturation_point["index"]
        sat_x = load_levels[saturation_index]
        sat_y = throughputs[saturation_index]
        ax1.scatter([sat_x], [sat_y], s=150, c='green', marker='*', zorder=10, 
                   label=f'Saturation Point ({sat_x} users)')
        ax1.axvline(x=sat_x, color='green', linestyle='--', alpha=0.5)
    
    inflection_point = analysis_results.get("response_time_inflection", {})
    if inflection_point.get("index") is not None:
        inflection_index = inflection_point["index"]
        inf_x = load_levels[inflection_index]
        inf_y = response_times[inflection_index]
        ax2.scatter([inf_x], [inf_y], s=150, c='purple', marker='*', zorder=10, 
                   label=f'RT Inflection Point ({inf_x} users)')
        ax1.axvline(x=inf_x, color='purple', linestyle='--', alpha=0.5)
    
    optimal_point = analysis_results.get("optimal_load_point", {})
    if optimal_point.get("load") is not None:
        opt_x = optimal_point["load"]
        opt_y_tp = optimal_point["throughput"]
        opt_y_rt = optimal_point["response_time"]
        ax1.scatter([opt_x], [opt_y_tp], s=150, c='orange', marker='*', zorder=10, 
                   label=f'Optimal Load ({opt_x} users)')
        ax1.axvline(x=opt_x, color='orange', linestyle='--', alpha=0.5)
    
    # Add title
    plt.title('Load Scalability Analysis', fontsize=16, fontweight='bold')
    
    # Add legends for both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Add explanatory text
    if analysis_results.get("has_degradation", False):
        degrad_text = "System shows performance degradation under high load."
    else:
        degrad_text = "No performance degradation detected within measured range."
    
    correlation = analysis_results.get("littles_law_correlation")
    if correlation:
        littles_text = f"Little's Law correlation: {correlation:.2f}"
        if correlation > 0.9:
            littles_text += " (Strong correlation)"
        elif correlation > 0.7:
            littles_text += " (Good correlation)"
        else:
            littles_text += " (Weak correlation)"
    else:
        littles_text = "Could not calculate Little's Law correlation."
        
    plt.figtext(0.15, 0.02, degrad_text + "\n" + littles_text, 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'load_scalability_analysis.png')
    plt.savefig(plot_path, dpi=150)
    plot_paths['load_scalability'] = plot_path
    
    if not show_plot:
        plt.close()
    
    # Create capacity model plot
    plt.figure(figsize=(10, 6))
    
    # Calculate theoretical max throughput based on available data
    max_throughput = np.max(throughputs)
    
    # Create load capacity model using simplified Universal Scalability Law
    # T(N) = T_max * N / (1 + α(N-1) + β*N*(N-1))
    # Where N is load level, α is contention, β is coherency
    
    try:
        # Define USL model for curve fitting
        def usl_model(N, Tmax, alpha, beta):
            return Tmax * N / (1 + alpha * (N - 1) + beta * N * (N - 1))
        
        # Fit model
        popt, _ = optimize.curve_fit(
            usl_model, np.array(load_levels), np.array(throughputs),
            bounds=([0, 0, 0], [max_throughput*2, 1, 1]),
            p0=[max_throughput, 0.1, 0.01]
        )
        
        # Extract parameters
        Tmax_fit, alpha_fit, beta_fit = popt
        
        # Create smooth curve with the fitted model
        x_smooth = np.linspace(min(load_levels)*0.9, max(load_levels)*2, 1000)
        y_smooth = usl_model(x_smooth, Tmax_fit, alpha_fit, beta_fit)
        
        # Plot actual data and model
        plt.scatter(load_levels, throughputs, s=80, color='blue', label='Actual Throughput')
        plt.plot(x_smooth, y_smooth, 'r-', linewidth=2, 
                 label=f'Capacity Model (α={alpha_fit:.4f}, β={beta_fit:.4f})')
        
        # Calculate theoretical maximum throughput
        if beta_fit > 0:
            N_max = np.sqrt((1 + alpha_fit) / beta_fit)
            T_max = usl_model(N_max, Tmax_fit, alpha_fit, beta_fit)
            plt.axhline(y=T_max, color='green', linestyle='--', 
                       label=f'Theoretical Max: {T_max:.2f} req/s')
            
            # Mark theoretical max point
            plt.scatter([N_max], [T_max], s=150, c='green', marker='*')
            
            model_info = {
                "Tmax_fit": Tmax_fit,
                "alpha_fit": alpha_fit,
                "beta_fit": beta_fit,
                "N_max": N_max,
                "T_max": T_max
            }
        else:
            model_info = {
                "Tmax_fit": Tmax_fit,
                "alpha_fit": alpha_fit,
                "beta_fit": beta_fit
            }
    except Exception as e:
        print(f"Warning: Could not fit capacity model: {str(e)}")
        model_info = {}
    
    # Add labels and title
    plt.title('Load Capacity Model', fontsize=16, fontweight='bold')
    plt.xlabel('Load Level (Users/Requests)', fontsize=12)
    plt.ylabel('Throughput (req/s)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Add model interpretation text
    if model_info:
        if 'alpha_fit' in model_info and 'beta_fit' in model_info:
            alpha = model_info["alpha_fit"]
            beta = model_info["beta_fit"]
            
            if beta > 0:
                if alpha > 0.1 and beta < 0.01:
                    interpretation = "System limited primarily by contention (resource conflicts)"
                elif alpha < 0.1 and beta > 0.01:
                    interpretation = "System limited primarily by coherency delays (coordination overhead)"
                elif alpha > 0.1 and beta > 0.01:
                    interpretation = "System limited by both contention and coherency delays"
                else:
                    interpretation = "System shows good scalability with load"
            else:
                interpretation = "Model suggests no coherency delays"
        else:
            interpretation = "Insufficient data for detailed model interpretation"
            
        plt.figtext(0.15, 0.02, interpretation, fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'load_capacity_model.png')
    plt.savefig(plot_path, dpi=150)
    plot_paths['capacity_model'] = plot_path
    
    if not show_plot:
        plt.close()
    
    return plot_paths


def interpret_load_scalability(analysis_results):
    """
    Interpret load scalability analysis results
    
    Args:
        analysis_results (dict): Results from analyze_load_scalability
        
    Returns:
        dict: Interpretation including insights and recommendations
    """
    if not analysis_results.get("success", False):
        return {
            "success": False,
            "message": "Insufficient data for reliable load scalability analysis"
        }
    
    # Extract key metrics
    saturation_load = analysis_results["saturation_point"]["load"]
    optimal_load = analysis_results["optimal_load_point"]["load"]
    has_degradation = analysis_results["has_degradation"]
    littles_law_correlation = analysis_results.get("littles_law_correlation", 0)
    
    # Build interpretation
    insights = []
    recommendations = []
    
    # Interpret saturation point
    insights.append(f"System throughput begins to saturate at approximately {saturation_load} users/requests.")
    
    # Interpret optimal load
    insights.append(f"The optimal load balancing throughput and response time is at {optimal_load} users/requests.")
    
    if optimal_load < saturation_load:
        insights.append("The optimal operating point occurs before throughput saturation, indicating good performance balance.")
    else:
        insights.append("The optimal operating point coincides with throughput saturation, suggesting throughput-focused optimization.")
    
    # Interpret performance degradation
    if has_degradation:
        insights.append("System shows performance degradation under high load, indicating potential resource exhaustion or bottlenecks.")
        recommendations.append("Investigate resource utilization (CPU, memory, connections) to identify the bottleneck causing degradation.")
    else:
        insights.append("No performance degradation detected within the measured load range, suggesting good stability.")
    
    # Interpret Little's Law correlation
    if littles_law_correlation is not None:
        if littles_law_correlation > 0.9:
            insights.append("Strong correlation with Little's Law, indicating predictable queueing behavior.")
        elif littles_law_correlation > 0.7:
            insights.append("Good correlation with Little's Law, with some deviation from ideal queueing.")
        else:
            insights.append("Weak correlation with Little's Law, suggesting complex system behavior beyond simple queueing.")
            recommendations.append("Investigate non-queueing factors like caching, connection pooling, or external dependencies.")
    
    # Add general recommendations
    if optimal_load < saturation_load * 0.7:
        recommendations.append(f"Consider limiting concurrency to around {optimal_load} for best balance of throughput and response time.")
    
    if not has_degradation:
        recommendations.append("Consider testing with higher load levels to identify the true system limits.")
    
    # Return formatted interpretation
    return {
        "success": True,
        "key_metrics": {
            "saturation_load": saturation_load,
            "optimal_load": optimal_load,
            "has_degradation": has_degradation
        },
        "insights": insights,
        "recommendations": recommendations
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Load Scalability Analysis')
    parser.add_argument('--load-levels', nargs='+', type=float, required=True,
                        help='Load levels (e.g., concurrent users, request rates)')
    parser.add_argument('--throughputs', nargs='+', type=float, required=True,
                        help='Corresponding throughput values')
    parser.add_argument('--response-times', nargs='+', type=float, required=True,
                        help='Corresponding response time values')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for plots')
    parser.add_argument('--show-plot', action='store_true',
                        help='Display the plots')
    
    args = parser.parse_args()
    
    if len(args.load_levels) != len(args.throughputs) or len(args.load_levels) != len(args.response_times):
        print("Error: Number of load levels must match number of throughputs and response times")
        exit(1)
    
    # Run analysis
    analysis_results = analyze_load_scalability(args.load_levels, args.throughputs, args.response_times)
    
    if analysis_results["success"]:
        # Generate plots
        plot_paths = plot_load_scalability(
            args.load_levels, args.throughputs, args.response_times,
            analysis_results, args.output_dir, args.show_plot
        )
        
        # Get interpretation
        interpretation = interpret_load_scalability(analysis_results)
        
        # Print results
        print(f"\nLoad Scalability Analysis Results:")
        print(f"Saturation point: {analysis_results['saturation_point']['load']} users/requests")
        print(f"Optimal load: {analysis_results['optimal_load_point']['load']} users/requests")
        
        print("\nInsights:")
        for insight in interpretation.get('insights', []):
            print(f"- {insight}")
        
        print("\nRecommendations:")
        for recommendation in interpretation.get('recommendations', []):
            print(f"- {recommendation}")
        
        if plot_paths:
            print(f"\nPlots saved to:")
            for name, path in plot_paths.items():
                print(f"- {name}: {path}")
    else:
        print(f"Analysis failed: {analysis_results.get('error', 'Unknown error')}")
