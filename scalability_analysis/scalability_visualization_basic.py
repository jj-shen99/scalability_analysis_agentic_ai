#!/usr/bin/env python3
"""
Basic Scalability Visualization Module

This module provides basic visualization functionality for scalability analysis:
- Throughput vs resource level plots
- Response time vs resource level plots
- Speedup vs resource level plots
- Basic comparative visualizations

Part of the modular scalability analysis framework
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# Set styling for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_throughput_vs_resource(resource_levels, throughputs, output_dir, show_plot=False):
    """
    Generate a plot of throughput vs resource level
    
    Args:
        resource_levels (list): Resource levels (e.g., number of nodes)
        throughputs (list): Corresponding throughput values
        output_dir (str): Directory to save the plot
        show_plot (bool): Whether to display the plot
        
    Returns:
        str: Path to the saved plot
    """
    plt.figure(figsize=(10, 6))
    
    # Create the scatter plot with points
    plt.scatter(resource_levels, throughputs, s=100, color='blue', label='Measured Throughput', zorder=5)
    
    # Add a trend line if we have enough points
    if len(resource_levels) > 1:
        z = np.polyfit(resource_levels, throughputs, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(resource_levels)*0.9, max(resource_levels)*1.1, 100)
        plt.plot(x_trend, p(x_trend), 'r--', label=f'Trend: {z[0]:.2f}x + {z[1]:.2f}')
        
        # Add the trend equation
        equation = f"y = {z[0]:.2f}x + {z[1]:.2f}"
        plt.annotate(equation, xy=(0.05, 0.95), xycoords='axes fraction', 
                     fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Add reference point annotations
    for i, (x, y) in enumerate(zip(resource_levels, throughputs)):
        plt.annotate(f"{y:.2f}", 
                     (x, y), 
                     textcoords="offset points", 
                     xytext=(0, 10), 
                     ha='center',
                     fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gray", alpha=0.8))
    
    # Customize the plot
    plt.title('Throughput vs Resource Level', fontsize=14, fontweight='bold')
    plt.xlabel('Resource Level', fontsize=12)
    plt.ylabel('Throughput (req/s)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(resource_levels)  # Force x-ticks to match resource levels
    
    # Ensure x-axis only shows integers
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    
    # Add legend
    plt.legend()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'throughput_vs_resource.png')
    plt.savefig(plot_path, dpi=150)
    
    if not show_plot:
        plt.close()
    
    return plot_path


def plot_response_time_vs_resource(resource_levels, response_times, output_dir, sla_threshold=None, show_plot=False):
    """
    Generate a plot of response time vs resource level
    
    Args:
        resource_levels (list): Resource levels (e.g., number of nodes)
        response_times (list): Corresponding response time values (ms)
        output_dir (str): Directory to save the plot
        sla_threshold (float, optional): SLA threshold to display on the plot
        show_plot (bool): Whether to display the plot
        
    Returns:
        str: Path to the saved plot
    """
    plt.figure(figsize=(10, 6))
    
    # Create the scatter plot with points
    plt.scatter(resource_levels, response_times, s=100, color='green', label='Measured Response Time', zorder=5)
    
    # Add a trend line if we have enough points
    if len(resource_levels) > 1:
        # Try using a power function (y = a*x^b) which often better models response time curves
        try:
            log_x = np.log(resource_levels)
            log_y = np.log(response_times)
            z = np.polyfit(log_x, log_y, 1)
            # For power function: y = exp(b) * x^m where b = intercept, m = slope
            b, m = z
            a = np.exp(b)
            
            x_trend = np.linspace(min(resource_levels)*0.9, max(resource_levels)*1.1, 100)
            plt.plot(x_trend, a * np.power(x_trend, m), 'r--', 
                     label=f'Trend: {a:.2f} × n^{m:.2f}')
        except:
            # Fall back to linear fit if power fit fails
            z = np.polyfit(resource_levels, response_times, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(resource_levels)*0.9, max(resource_levels)*1.1, 100)
            plt.plot(x_trend, p(x_trend), 'r--', label=f'Trend: {z[0]:.2f}x + {z[1]:.2f}')
    
    # Add reference point annotations
    for i, (x, y) in enumerate(zip(resource_levels, response_times)):
        plt.annotate(f"{y:.2f} ms", 
                     (x, y), 
                     textcoords="offset points", 
                     xytext=(0, 10), 
                     ha='center',
                     fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gray", alpha=0.8))
    
    # Add SLA threshold if provided
    if sla_threshold is not None:
        plt.axhline(y=sla_threshold, color='red', linestyle='-', 
                   label=f'SLA Threshold: {sla_threshold} ms')
        
        # Check if any response time exceeds the threshold
        violations = [rt > sla_threshold for rt in response_times]
        if any(violations):
            plt.fill_between(
                [min(resource_levels)*0.9, max(resource_levels)*1.1], 
                sla_threshold, 
                max(response_times)*1.1,
                color='red', alpha=0.1, label='SLA Violation Zone'
            )
    
    # Customize the plot
    plt.title('Response Time vs Resource Level', fontsize=14, fontweight='bold')
    plt.xlabel('Resource Level', fontsize=12)
    plt.ylabel('Average Response Time (ms)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(resource_levels)  # Force x-ticks to match resource levels
    
    # Ensure x-axis only shows integers
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    
    # Add legend
    plt.legend()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'response_time_vs_resource.png')
    plt.savefig(plot_path, dpi=150)
    
    if not show_plot:
        plt.close()
    
    return plot_path


def plot_speedup_vs_resource(resource_levels, speedups, output_dir, show_plot=False):
    """
    Generate a plot of speedup vs resource level
    
    Args:
        resource_levels (list): Resource levels (e.g., number of nodes)
        speedups (list): Corresponding speedup values
        output_dir (str): Directory to save the plot
        show_plot (bool): Whether to display the plot
        
    Returns:
        str: Path to the saved plot
    """
    plt.figure(figsize=(10, 6))
    
    # Create the scatter plot with points
    plt.scatter(resource_levels, speedups, s=100, color='blue', label='Actual Speedup', zorder=5)
    
    # Add ideal linear speedup line
    max_resource = max(resource_levels) * 1.1
    ideal_x = np.array([min(resource_levels), max_resource])
    ideal_y = ideal_x / min(resource_levels)  # Assuming first resource level is baseline
    plt.plot(ideal_x, ideal_y, 'k--', label='Ideal Linear Speedup')
    
    # Add efficiency lines (25%, 50%, 75%)
    for efficiency in [0.75, 0.5, 0.25]:
        eff_y = ideal_x / min(resource_levels) * efficiency
        plt.plot(ideal_x, eff_y, 'k:', alpha=0.3, label=f'{efficiency*100:.0f}% Efficiency')
    
    # Add reference point annotations
    for i, (x, y) in enumerate(zip(resource_levels, speedups)):
        # Calculate efficiency
        ideal = x / min(resource_levels)
        efficiency = y / ideal if ideal > 0 else 0
        
        plt.annotate(f"{y:.2f}x ({efficiency:.1%})", 
                     (x, y), 
                     textcoords="offset points", 
                     xytext=(0, 10), 
                     ha='center',
                     fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gray", alpha=0.8))
    
    # Customize the plot
    plt.title('Speedup vs Resource Level', fontsize=14, fontweight='bold')
    plt.xlabel('Resource Level', fontsize=12)
    plt.ylabel('Speedup Factor', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(resource_levels)  # Force x-ticks to match resource levels
    
    # Ensure x-axis only shows integers
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    
    # Add legend
    plt.legend()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'speedup_vs_resource.png')
    plt.savefig(plot_path, dpi=150)
    
    if not show_plot:
        plt.close()
    
    return plot_path


def plot_comparative_throughput(dfs, configs, output_dir, show_plot=False):
    """
    Generate a comparative plot of throughput across different configurations
    
    Args:
        dfs (list): List of dataframes with timestamp and elapsed data
        configs (list): List of configuration labels
        output_dir (str): Directory to save the plot
        show_plot (bool): Whether to display the plot
        
    Returns:
        str: Path to the saved plot
    """
    if not dfs or len(dfs) < 2:
        print("Error: At least two datasets are required for comparison")
        return None
    
    plt.figure(figsize=(12, 7))
    
    # Calculate and plot throughput for each configuration
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink']
    throughputs = []
    
    for i, (df, config) in enumerate(zip(dfs, configs)):
        if 'timeStamp' not in df.columns or df.empty:
            continue
            
        # Calculate throughput
        start_time = df['timeStamp'].min()
        end_time = df['timeStamp'].max()
        duration_seconds = (end_time - start_time).total_seconds()
        
        if duration_seconds > 0:
            throughput = len(df) / duration_seconds
            throughputs.append(throughput)
            
            # Create bar
            color = colors[i % len(colors)]
            plt.bar(i, throughput, color=color, alpha=0.7, label=f"{config}: {throughput:.2f} req/s")
    
    # Add labels
    plt.title('Comparative Throughput Analysis', fontsize=14, fontweight='bold')
    plt.ylabel('Throughput (requests/sec)', fontsize=12)
    plt.xticks(range(len(configs)), configs, rotation=45)
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add throughput values on top of bars
    for i, throughput in enumerate(throughputs):
        plt.text(i, throughput + max(throughputs) * 0.02, f"{throughput:.2f}", 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add speedup annotations
    if len(throughputs) > 1:
        baseline = throughputs[0]
        for i, throughput in enumerate(throughputs[1:], 1):
            speedup = throughput / baseline if baseline > 0 else 0
            plt.text(i, throughput / 2, f"{speedup:.2f}x", 
                     ha='center', va='center', fontsize=12, fontweight='bold',
                     color='white')
    
    plt.tight_layout()
    
    # Add legend
    plt.legend()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'comparative_throughput.png')
    plt.savefig(plot_path, dpi=150)
    
    if not show_plot:
        plt.close()
    
    return plot_path


def plot_response_time_distribution(dfs, configs, output_dir, show_plot=False):
    """
    Generate a plot comparing response time distributions across configurations
    
    Args:
        dfs (list): List of dataframes with elapsed data
        configs (list): List of configuration labels
        output_dir (str): Directory to save the plot
        show_plot (bool): Whether to display the plot
        
    Returns:
        str: Path to the saved plot
    """
    if not dfs or len(dfs) < 1:
        print("Error: At least one dataset is required")
        return None
    
    plt.figure(figsize=(12, 7))
    
    # Plot histogram/kde for each configuration
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink']
    
    for i, (df, config) in enumerate(zip(dfs, configs)):
        if 'elapsed' not in df.columns or df.empty:
            continue
            
        color = colors[i % len(colors)]
        
        # Create KDE plot
        sns.kdeplot(df['elapsed'], label=f"{config} (avg: {df['elapsed'].mean():.2f}ms)", 
                   color=color, alpha=0.7, fill=True)
    
    # Add vertical lines for averages
    for i, (df, config) in enumerate(zip(dfs, configs)):
        if 'elapsed' not in df.columns or df.empty:
            continue
            
        color = colors[i % len(colors)]
        avg_rt = df['elapsed'].mean()
        
        plt.axvline(x=avg_rt, color=color, linestyle='--')
        plt.text(avg_rt, plt.gca().get_ylim()[1]*0.9*(0.8**i), 
                f"{avg_rt:.2f}ms", 
                color=color, fontweight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=color, alpha=0.8))
    
    # Customize the plot
    plt.title('Response Time Distribution Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Response Time (ms)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Add legend
    plt.legend()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'response_time_distribution.png')
    plt.savefig(plot_path, dpi=150)
    
    if not show_plot:
        plt.close()
    
    return plot_path


def create_basic_scalability_plots(resource_levels, throughputs, response_times, output_dir, show_plots=False):
    """
    Create all basic scalability plots
    
    Args:
        resource_levels (list): Resource levels (e.g., number of nodes)
        throughputs (list): Corresponding throughput values
        response_times (list): Corresponding response time values
        output_dir (str): Directory to save the plots
        show_plots (bool): Whether to display the plots
        
    Returns:
        dict: Paths to saved plots
    """
    plot_paths = {}
    
    # Create throughput plot
    throughput_path = plot_throughput_vs_resource(resource_levels, throughputs, output_dir, show_plots)
    plot_paths['throughput'] = throughput_path
    
    # Create response time plot
    rt_path = plot_response_time_vs_resource(resource_levels, response_times, output_dir, show_plots)
    plot_paths['response_time'] = rt_path
    
    # Calculate speedups
    baseline_throughput = throughputs[0] if throughputs else 1
    speedups = [t / baseline_throughput for t in throughputs]
    
    # Create speedup plot
    speedup_path = plot_speedup_vs_resource(resource_levels, speedups, output_dir, show_plots)
    plot_paths['speedup'] = speedup_path
    
    return plot_paths


def plot_efficiency_heatmap(resource_levels, throughputs, output_dir, show_plot=False):
    """
    Generate a heatmap visualizing scalability efficiency across different resource levels
    
    Args:
        resource_levels (list): Resource levels (e.g., number of nodes)
        throughputs (list): Corresponding throughput values
        output_dir (str): Directory to save the plot
        show_plot (bool): Whether to display the plot
        
    Returns:
        str: Path to the saved plot
    """
    if len(resource_levels) < 2:
        print("Error: At least two data points are needed for efficiency heatmap")
        return None
    
    plt.figure(figsize=(10, 8))
    
    # Calculate speedup matrix and efficiency matrix
    n = len(resource_levels)
    speedup_matrix = np.zeros((n, n))
    efficiency_matrix = np.zeros((n, n))
    
    # Fill matrices
    for i in range(n):
        for j in range(n):
            if i == j:
                speedup_matrix[i, j] = 1.0
                efficiency_matrix[i, j] = 1.0
            elif i < j:  # Only calculate for resource increase
                speedup = throughputs[j] / throughputs[i]
                speedup_matrix[i, j] = speedup
                
                # Calculate efficiency: speedup / resource_ratio
                resource_ratio = resource_levels[j] / resource_levels[i]
                efficiency_matrix[i, j] = speedup / resource_ratio
    
    # Create heatmap for efficiency
    sns.heatmap(efficiency_matrix, annot=True, fmt=".2f", cmap="RdYlGn", 
                vmin=0, vmax=1.0, square=True,
                xticklabels=resource_levels, yticklabels=resource_levels)
    
    # Add labels and title
    plt.title("Scalability Efficiency Heatmap", fontsize=14, fontweight="bold")
    plt.xlabel("Target Resource Level", fontsize=12)
    plt.ylabel("Baseline Resource Level", fontsize=12)
    
    # Add description
    plt.figtext(0.5, 0.01, "Values indicate efficiency: speedup / resource ratio\n" +
                "1.0 = perfect scaling, <1.0 = sub-linear scaling, >1.0 = super-linear scaling",
                ha="center", fontsize=10, bbox={"facecolor":"aliceblue", "alpha":0.5, "pad":5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'efficiency_heatmap.png')
    plt.savefig(plot_path, dpi=150)
    
    if not show_plot:
        plt.close()
    
    return plot_path


def plot_scalability_efficiency(resource_levels, throughputs, output_dir, model_params=None, show_plot=False):
    """
    Plot scalability efficiency analysis showing how efficiency changes across resource levels
    
    Args:
        resource_levels (list): Resource levels (e.g., number of nodes)
        throughputs (list): Corresponding throughput values
        output_dir (str): Directory to save the plot
        model_params (dict, optional): Scalability model parameters for overlay
        show_plot (bool): Whether to display the plot
        
    Returns:
        str: Path to the saved plot
    """
    if len(resource_levels) < 2:
        print("Error: At least two data points are needed for efficiency analysis")
        return None
    
    plt.figure(figsize=(10, 6))
    
    # Calculate baseline efficiency
    baseline_throughput = throughputs[0]
    baseline_resource = resource_levels[0]
    
    # Calculate speedups and efficiencies
    speedups = [t / baseline_throughput for t in throughputs]
    resource_ratios = [r / baseline_resource for r in resource_levels]
    efficiencies = [s / r for s, r in zip(speedups, resource_ratios)]
    
    # Plot efficiency vs. resource level
    plt.plot(resource_levels, efficiencies, 'o-', color='green', linewidth=2, markersize=8)
    
    # Add reference line at 100% efficiency
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label="Perfect Scaling (100%)")
    
    # Add reference lines at 75% and 50% efficiency
    plt.axhline(y=0.75, color='orange', linestyle=':', alpha=0.5, label="75% Efficiency")
    plt.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, label="50% Efficiency")
    
    # Add efficiency values to data points
    for i, (x, y) in enumerate(zip(resource_levels, efficiencies)):
        plt.annotate(f"{y:.2%}", 
                     (x, y), 
                     textcoords="offset points", 
                     xytext=(0, 10), 
                     ha='center',
                     fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gray", alpha=0.8))
    
    # Add model curves if parameters are provided
    if model_params and 'amdahl' in model_params:
        from scalability_models import amdahls_law
        p = model_params['amdahl']
        x_model = np.linspace(min(resource_levels), max(resource_levels)*1.5, 100)
        
        # Calculate Amdahl's Law efficiency
        amdahl_speedups = [amdahls_law(p, r / baseline_resource) for r in x_model]
        amdahl_efficiency = [s / (r / baseline_resource) for s, r in zip(amdahl_speedups, x_model)]
        
        plt.plot(x_model, amdahl_efficiency, 'b--', alpha=0.6, 
                 label=f"Amdahl's Law Model (p={p:.2f})")
    
    # Customize the plot
    plt.title('Scalability Efficiency vs Resource Level', fontsize=14, fontweight='bold')
    plt.xlabel('Resource Level', fontsize=12)
    plt.ylabel('Efficiency (Speedup/Resource Ratio)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(resource_levels)
    
    # Set y-axis to show percentages
    plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    
    # Ensure x-axis only shows integers
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add legend
    plt.legend(loc='best')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'scalability_efficiency.png')
    plt.savefig(plot_path, dpi=150)
    
    if not show_plot:
        plt.close()
    
    return plot_path


def plot_cost_efficiency(resource_levels, throughputs, output_dir, cost_model=None, show_plot=False):
    """
    Generate a cost efficiency analysis plot
    
    Args:
        resource_levels (list): Resource levels (e.g., number of nodes)
        throughputs (list): Corresponding throughput values
        output_dir (str): Directory to save the plot
        cost_model (func, optional): Function to calculate cost based on resource level
        show_plot (bool): Whether to display the plot
        
    Returns:
        str: Path to the saved plot
    """
    plt.figure(figsize=(12, 8))
    
    # Use a default linear cost model if none provided
    if cost_model is None:
        cost_model = lambda x: x  # Default: cost is proportional to resource level
    
    # Calculate costs
    costs = [cost_model(r) for r in resource_levels]
    
    # Calculate throughput per cost unit
    throughput_per_cost = [t / c if c > 0 else 0 for t, c in zip(throughputs, costs)]
    
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Throughput vs Cost
    ax1.plot(costs, throughputs, 'o-', color='blue', linewidth=2, markersize=8)
    
    # Add reference point annotations to first plot
    for i, (x, y, r) in enumerate(zip(costs, throughputs, resource_levels)):
        ax1.annotate(f"R{r}\n({y:.0f} req/s)", 
                    (x, y), 
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center',
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gray", alpha=0.8))
    
    # Add trendline to first plot
    if len(costs) > 1:
        z = np.polyfit(costs, throughputs, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(costs)*0.9, max(costs)*1.1, 100)
        ax1.plot(x_trend, p(x_trend), 'r--', alpha=0.7)
    
    ax1.set_title('Throughput vs. Cost', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Cost Units', fontsize=12)
    ax1.set_ylabel('Throughput (req/s)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Throughput per Cost Unit
    bars = ax2.bar(resource_levels, throughput_per_cost, color='green', alpha=0.7)
    
    # Add value labels on top of bars
    for bar, value in zip(bars, throughput_per_cost):
        ax2.annotate(f"{value:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gray", alpha=0.8))
    
    # Find and highlight the most cost-effective configuration
    most_efficient_idx = np.argmax(throughput_per_cost)
    bars[most_efficient_idx].set_color('darkgreen')
    bars[most_efficient_idx].set_alpha(1.0)
    
    ax2.set_title('Cost Efficiency Analysis', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Resource Level', fontsize=12)
    ax2.set_ylabel('Throughput per Cost Unit', fontsize=12)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add annotation for most efficient configuration
    ax2.annotate(f"Most Cost Efficient: {resource_levels[most_efficient_idx]}",
                xy=(resource_levels[most_efficient_idx], throughput_per_cost[most_efficient_idx]),
                xytext=(0, 20),
                textcoords="offset points",
                ha='center',
                fontsize=10,
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'cost_efficiency_analysis.png')
    plt.savefig(plot_path, dpi=150)
    
    if not show_plot:
        plt.close()
    
    return plot_path


def create_advanced_scalability_plots(resource_levels, throughputs, output_dir, model_params=None, show_plots=False):
    """
    Create all advanced scalability analysis plots
    
    Args:
        resource_levels (list): Resource levels (e.g., number of nodes)
        throughputs (list): Corresponding throughput values
        output_dir (str): Directory to save the plots
        model_params (dict, optional): Scalability model parameters
        show_plots (bool): Whether to display the plots
        
    Returns:
        dict: Paths to saved plots
    """
    plot_paths = {}
    
    if len(resource_levels) >= 2:
        # Scalability efficiency plot
        efficiency_path = plot_scalability_efficiency(
            resource_levels, throughputs, output_dir, model_params, show_plots
        )
        plot_paths['efficiency'] = efficiency_path
        
        # Efficiency heatmap
        if len(resource_levels) >= 3:  # More interesting with at least 3 points
            heatmap_path = plot_efficiency_heatmap(
                resource_levels, throughputs, output_dir, show_plots
            )
            plot_paths['heatmap'] = heatmap_path
        
        # Cost efficiency analysis
        cost_path = plot_cost_efficiency(
            resource_levels, throughputs, output_dir, None, show_plots
        )
        plot_paths['cost_efficiency'] = cost_path
    
    return plot_paths


# Main entry point for direct script execution
if __name__ == "__main__":
    import argparse
    from scalability_core import analyze_jtl, create_output_dir
    
    parser = argparse.ArgumentParser(description='Scalability Visualization Module')
    parser.add_argument('--files', nargs='+', required=True, help='JTL files to analyze')
    parser.add_argument('--levels', nargs='+', type=int, required=True, 
                        help='Resource levels corresponding to files')
    parser.add_argument('--output-dir', type=str, help='Output directory for plots')
    parser.add_argument('--advanced', action='store_true', help='Generate advanced plots')
    parser.add_argument('--show-plots', action='store_true', help='Display plots')
    
    args = parser.parse_args()
    
    if len(args.files) != len(args.levels):
        print("Error: Number of files must match number of resource levels")
        exit(1)
    
    # Create output directory
    output_dir = args.output_dir or create_output_dir()
    
    # Analyze files
    resource_levels = []
    throughputs = []
    response_times = []
    
    for file_path, level in zip(args.files, args.levels):
        metrics = analyze_jtl(file_path)
        if metrics:
            resource_levels.append(level)
            throughputs.append(metrics['throughput'])
            response_times.append(metrics['avg_response_time'])
    
    # Create plots
    if resource_levels:
        # Basic plots
        basic_plots = create_basic_scalability_plots(
            resource_levels, throughputs, response_times, output_dir, args.show_plots
        )
        print(f"Basic plots saved to {output_dir}:")
        for name, path in basic_plots.items():
            print(f"  {name}: {os.path.basename(path)}")
            
        # Advanced plots if requested
        if args.advanced:
            advanced_plots = create_advanced_scalability_plots(
                resource_levels, throughputs, output_dir, None, args.show_plots
            )
            print(f"Advanced plots saved to {output_dir}:")
            for name, path in advanced_plots.items():
                print(f"  {name}: {os.path.basename(path)}")
    else:
        print("Error: No valid data to plot")
