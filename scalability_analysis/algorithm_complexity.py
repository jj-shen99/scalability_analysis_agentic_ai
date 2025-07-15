#!/usr/bin/env python3
"""
Algorithm Complexity Analysis Module

This module provides functions to analyze and identify the algorithmic complexity
of a system based on performance data across different load levels.

Author: JJ Shen
"""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator

# Define common complexity functions for fitting
def constant_time(n, a):
    """O(1) - Constant time complexity"""
    return a * np.ones_like(n)

def logarithmic_time(n, a, b):
    """O(log n) - Logarithmic time complexity"""
    return a * np.log(n) + b

def linear_time(n, a, b):
    """O(n) - Linear time complexity"""
    return a * n + b

def linearithmic_time(n, a, b):
    """O(n log n) - Linearithmic time complexity"""
    return a * n * np.log(n) + b

def quadratic_time(n, a, b):
    """O(n²) - Quadratic time complexity"""
    return a * n**2 + b

def cubic_time(n, a, b):
    """O(n³) - Cubic time complexity"""
    return a * n**3 + b

def exponential_time(n, a, b, c):
    """O(c^n) - Exponential time complexity"""
    return a * c**n + b


def analyze_algorithm_complexity(load_sizes, execution_times):
    """
    Analyze performance data to determine the likely algorithmic complexity
    
    Args:
        load_sizes (list): Input sizes or load levels
        execution_times (list): Corresponding execution times
        
    Returns:
        dict: Analysis results including best fit model, parameters, and error metrics
    """
    if len(load_sizes) < 3:
        return {
            "success": False,
            "error": "Insufficient data points. At least 3 points needed for reliable complexity analysis."
        }
    
    # Convert to numpy arrays for processing
    x = np.array(load_sizes)
    y = np.array(execution_times)
    
    # Define models to test
    models = {
        "O(1)": (constant_time, 1),
        "O(log n)": (logarithmic_time, 2),
        "O(n)": (linear_time, 2),
        "O(n log n)": (linearithmic_time, 2),
        "O(n²)": (quadratic_time, 2),
        "O(n³)": (cubic_time, 2),
        "O(c^n)": (exponential_time, 3)
    }
    
    results = {}
    best_error = float('inf')
    best_model = None
    
    # Try fitting each model
    for name, (func, num_params) in models.items():
        try:
            # Initial parameter guesses
            if num_params == 1:
                p0 = [1]
            elif num_params == 2:
                p0 = [1, 0]
            else:
                p0 = [1, 0, 2]  # For exponential, assume base 2 initially
            
            # Fit the model
            params, covariance = optimize.curve_fit(func, x, y, p0=p0, maxfev=10000)
            
            # Calculate error metrics
            y_pred = func(x, *params)
            residuals = y - y_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            rmse = np.sqrt(np.mean(residuals**2))
            
            # Store results
            results[name] = {
                "params": params,
                "r_squared": r_squared,
                "rmse": rmse,
                "function": func
            }
            
            # Track best model
            if rmse < best_error:
                best_error = rmse
                best_model = name
                
        except Exception as e:
            results[name] = {
                "error": str(e)
            }
    
    # Return analysis results
    return {
        "success": True,
        "best_fit": best_model,
        "models": results,
        "data_points": len(load_sizes)
    }


def plot_algorithm_complexity(load_sizes, execution_times, analysis_results, output_dir, show_plot=False):
    """
    Generate plot showing algorithm complexity analysis with curve fitting
    
    Args:
        load_sizes (list): Input sizes or load levels
        execution_times (list): Corresponding execution times
        analysis_results (dict): Results from analyze_algorithm_complexity
        output_dir (str): Directory to save the plot
        show_plot (bool): Whether to display the plot
        
    Returns:
        str: Path to saved plot
    """
    if not analysis_results.get("success", False):
        print("Error: Cannot generate algorithm complexity plot due to analysis failure")
        return None
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Convert to numpy arrays
    x = np.array(load_sizes)
    y = np.array(execution_times)
    
    # Plot actual data points
    plt.scatter(x, y, s=80, color='blue', label='Actual Data', zorder=5)
    
    # Create smooth x values for curves
    x_smooth = np.linspace(min(x)*0.9, max(x)*1.1, 1000)
    
    # Plot the best fit model
    best_model = analysis_results["best_fit"]
    best_params = analysis_results["models"][best_model]["params"]
    best_func = analysis_results["models"][best_model]["function"]
    best_r2 = analysis_results["models"][best_model]["r_squared"]
    
    y_best = best_func(x_smooth, *best_params)
    plt.plot(x_smooth, y_best, 'r-', linewidth=2, 
             label=f'{best_model} (Best Fit, R² = {best_r2:.3f})', zorder=4)
    
    # Plot other top models for comparison (top 2 besides the best)
    models_by_r2 = sorted(
        [(name, model["r_squared"]) for name, model in analysis_results["models"].items() 
         if "r_squared" in model and name != best_model],
        key=lambda x: x[1],
        reverse=True
    )
    
    colors = ['g--', 'm-.', 'c:']
    for i, (name, r2) in enumerate(models_by_r2[:2]):  # Top 2 alternatives
        if i < len(colors) and "function" in analysis_results["models"][name]:
            func = analysis_results["models"][name]["function"]
            params = analysis_results["models"][name]["params"]
            y_model = func(x_smooth, *params)
            plt.plot(x_smooth, y_model, colors[i], alpha=0.7, linewidth=1.5,
                    label=f'{name} (R² = {r2:.3f})', zorder=3-i)
    
    # Add labels and title
    plt.title('Algorithm Complexity Analysis', fontsize=16, fontweight='bold')
    plt.xlabel('Input Size / Load Level', fontsize=14)
    plt.ylabel('Execution Time', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend(loc='best', fontsize=12)
    
    # Add annotations for actual data points
    for i, (x_val, y_val) in enumerate(zip(x, y)):
        plt.annotate(f"({x_val}, {y_val:.2f})", 
                     (x_val, y_val), 
                     textcoords="offset points",
                     xytext=(0, 10), 
                     ha='center',
                     fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gray", alpha=0.8))
    
    # Add complexity interpretation text box
    complexity_notes = {
        "O(1)": "Constant time - Independent of input size",
        "O(log n)": "Logarithmic time - Typically seen in binary search, divide and conquer",
        "O(n)": "Linear time - Time grows linearly with input size",
        "O(n log n)": "Linearithmic time - Typical of efficient sorting algorithms",
        "O(n²)": "Quadratic time - Nested loops, comparison-based operations",
        "O(n³)": "Cubic time - Triple nested loops, matrix operations",
        "O(c^n)": "Exponential time - Combinatorial problems, brute force"
    }
    
    textbox_content = f"Best fit model: {best_model}\n{complexity_notes.get(best_model, '')}"
    plt.figtext(0.15, 0.02, textbox_content, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'algorithm_complexity_analysis.png')
    plt.savefig(plot_path, dpi=150)
    
    if not show_plot:
        plt.close()
    
    return plot_path


def interpret_algorithm_complexity(analysis_results):
    """
    Interpret algorithm complexity analysis results
    
    Args:
        analysis_results (dict): Results from analyze_algorithm_complexity
        
    Returns:
        dict: Interpretation including insights and recommendations
    """
    if not analysis_results.get("success", False):
        return {
            "success": False,
            "message": "Insufficient data for reliable complexity analysis"
        }
    
    best_model = analysis_results["best_fit"]
    best_r2 = analysis_results["models"][best_model]["r_squared"]
    
    # Interpret the best-fit model
    interpretations = {
        "O(1)": {
            "explanation": "The system appears to have constant-time complexity, which is independent of input size.",
            "implications": "This suggests the system has excellent scalability as performance remains stable regardless of load.",
            "recommendations": "This is ideal behavior; no optimization needed for algorithm complexity."
        },
        "O(log n)": {
            "explanation": "The system appears to have logarithmic time complexity.",
            "implications": "This indicates very good scalability, typical of efficient search algorithms or data structures.",
            "recommendations": "This is already very efficient; focus on other aspects of optimization if needed."
        },
        "O(n)": {
            "explanation": "The system appears to have linear time complexity.",
            "implications": "Performance degrades linearly with input size, which is acceptable for many applications.",
            "recommendations": "Consider if any operations can be optimized to sub-linear time (e.g., using indexing or caching)."
        },
        "O(n log n)": {
            "explanation": "The system appears to have linearithmic time complexity.",
            "implications": "This is typical of efficient sorting algorithms or divide-and-conquer approaches.",
            "recommendations": "This is generally good. Review if sorting operations are necessary or if they can be optimized."
        },
        "O(n²)": {
            "explanation": "The system appears to have quadratic time complexity.",
            "implications": "Performance will degrade significantly with larger inputs, suggesting nested loops or comparisons.",
            "recommendations": "Look for nested loops or O(n²) operations that could be optimized with more efficient algorithms."
        },
        "O(n³)": {
            "explanation": "The system appears to have cubic time complexity.",
            "implications": "Performance will degrade very rapidly with input size, suggesting triple nested operations.",
            "recommendations": "Urgent optimization needed. Look for nested loops and consider fundamental algorithm changes."
        },
        "O(c^n)": {
            "explanation": "The system appears to have exponential time complexity.",
            "implications": "Performance will degrade extremely rapidly, making the system unusable for large inputs.",
            "recommendations": "Critical optimization needed. Consider dynamic programming, memoization, or approximate algorithms."
        }
    }
    
    # Get interpretation for best model
    interpretation = interpretations.get(best_model, {
        "explanation": "The system's complexity doesn't clearly match a standard algorithmic pattern.",
        "implications": "Custom or hybrid algorithm behavior observed.",
        "recommendations": "Detailed code review needed to identify specific performance patterns."
    })
    
    # Add confidence assessment
    if best_r2 > 0.95:
        confidence = "Very high confidence in the model fit."
    elif best_r2 > 0.9:
        confidence = "High confidence in the model fit."
    elif best_r2 > 0.8:
        confidence = "Moderate confidence in the model fit."
    elif best_r2 > 0.7:
        confidence = "Low confidence in the model fit. Results should be treated as preliminary."
    else:
        confidence = "Very low confidence in the model fit. Consider gathering more data points."
    
    return {
        "success": True,
        "best_model": best_model,
        "confidence": confidence,
        "explanation": interpretation["explanation"],
        "implications": interpretation["implications"],
        "recommendations": interpretation["recommendations"],
        "r_squared": best_r2
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Algorithm Complexity Analysis')
    parser.add_argument('--load-sizes', nargs='+', type=float, required=True,
                        help='Load sizes or input sizes')
    parser.add_argument('--times', nargs='+', type=float, required=True,
                        help='Execution times corresponding to load sizes')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for plots')
    parser.add_argument('--show-plot', action='store_true',
                        help='Display the plot')
    
    args = parser.parse_args()
    
    if len(args.load_sizes) != len(args.times):
        print("Error: Number of load sizes must match number of execution times")
        exit(1)
    
    # Run analysis
    analysis_results = analyze_algorithm_complexity(args.load_sizes, args.times)
    
    if analysis_results["success"]:
        # Generate plot
        plot_path = plot_algorithm_complexity(
            args.load_sizes, args.times, analysis_results, args.output_dir, args.show_plot
        )
        
        # Get interpretation
        interpretation = interpret_algorithm_complexity(analysis_results)
        
        # Print results
        print(f"\nAlgorithm Complexity Analysis Results:")
        print(f"Best fitting model: {analysis_results['best_fit']}")
        print(f"Confidence: {interpretation['confidence']}")
        print(f"\nExplanation: {interpretation['explanation']}")
        print(f"Implications: {interpretation['implications']}")
        print(f"Recommendations: {interpretation['recommendations']}")
        
        if plot_path:
            print(f"\nPlot saved to: {plot_path}")
    else:
        print(f"Analysis failed: {analysis_results.get('error', 'Unknown error')}")
