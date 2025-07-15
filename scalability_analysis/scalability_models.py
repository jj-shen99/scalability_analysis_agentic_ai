import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import os

def amdahls_law(p, n):
    """
    Calculate speedup according to Amdahl's Law
    
    Args:
        p (float): Fraction of the program that is parallelizable (between 0 and 1)
        n (int or list): Number of processors/resources
        
    Returns:
        float or list: Speedup factor(s)
    """
    # If n is a single value
    if isinstance(n, (int, float)):
        return 1 / ((1 - p) + p / n)
    
    # If n is a list/array
    return [1 / ((1 - p) + p / i) for i in n]

def fit_amdahls_law(resource_levels, speedups):
    """
    Fit Amdahl's Law to observed data and return the parallelizable fraction
    
    Args:
        resource_levels (list): List of resource levels used in tests
        speedups (list): List of observed speedups
        
    Returns:
        tuple: (p, residual error)
            - p: Estimated parallelizable fraction
            - residual error: Sum of squared errors between model and observations
    """
    def error_function(p):
        predicted = amdahls_law(p, resource_levels)
        return sum((np.array(predicted) - np.array(speedups)) ** 2)
    
    # Optimization to find the best p value (bounded between 0 and 1)
    result = optimize.minimize_scalar(error_function, bounds=(0, 1), method='bounded')
    
    # Return the optimized p value and the residual error
    return result.x, result.fun

def gustafsons_law(p, n):
    """
    Calculate speedup according to Gustafson's Law
    
    Args:
        p (float): Fraction of the program that is parallelizable (between 0 and 1)
        n (int or list): Number of processors/resources
        
    Returns:
        float or list: Speedup factor(s)
    """
    # If n is a single value
    if isinstance(n, (int, float)):
        return (1 - p) + n * p
    
    # If n is a list/array
    return [(1 - p) + i * p for i in n]

def fit_gustafsons_law(resource_levels, speedups):
    """
    Fit Gustafson's Law to observed data and return the parallelizable fraction
    
    Args:
        resource_levels (list): List of resource levels used in tests
        speedups (list): List of observed speedups
        
    Returns:
        tuple: (p, residual error)
            - p: Estimated parallelizable fraction
            - residual error: Sum of squared errors between model and observations
    """
    def error_function(p):
        predicted = gustafsons_law(p, resource_levels)
        return sum((np.array(predicted) - np.array(speedups)) ** 2)
    
    # Optimization to find the best p value (bounded between 0 and 1)
    result = optimize.minimize_scalar(error_function, bounds=(0, 1), method='bounded')
    
    # Return the optimized p value and the residual error
    return result.x, result.fun

def universal_scalability_law(sigma, kappa, n):
    """
    Calculate speedup according to Universal Scalability Law
    
    Args:
        sigma (float): Contention/serialization factor (between 0 and 1)
        kappa (float): Coherency delay factor (between 0 and 1)
        n (int or list): Number of processors/resources
        
    Returns:
        float or list: Speedup factor(s)
    """
    # If n is a single value
    if isinstance(n, (int, float)):
        return n / (1 + sigma * (n - 1) + kappa * n * (n - 1))
    
    # If n is a list/array
    return [i / (1 + sigma * (i - 1) + kappa * i * (i - 1)) for i in n]

def fit_universal_scalability_law(resource_levels, speedups):
    """
    Fit Universal Scalability Law to observed data and return parameters
    
    Args:
        resource_levels (list): List of resource levels used in tests
        speedups (list): List of observed speedups
        
    Returns:
        tuple: (sigma, kappa, residual error)
            - sigma: Estimated contention/serialization factor
            - kappa: Estimated coherency delay factor
            - residual error: Sum of squared errors between model and observations
    """
    def error_function(params):
        sigma, kappa = params
        predicted = universal_scalability_law(sigma, kappa, resource_levels)
        return sum((np.array(predicted) - np.array(speedups)) ** 2)
    
    # Initial guess for sigma and kappa
    initial_guess = [0.1, 0.01]
    
    # Bounds for sigma and kappa (both should be between 0 and 1)
    bounds = ((0, 1), (0, 1))
    
    # Optimization to find the best sigma and kappa values
    result = optimize.minimize(error_function, initial_guess, bounds=bounds, method='L-BFGS-B')
    
    # Return the optimized sigma and kappa values, along with the residual error
    return result.x[0], result.x[1], result.fun

def plot_scalability_models(resource_levels, actual_speedups, output_dir, models=None, show_plots=False):
    """
    Plot comparisons of actual speedup vs. theoretical models
    
    Args:
        resource_levels (list): List of resource levels used in tests
        actual_speedups (list): List of observed speedups
        output_dir (str): Directory to save plots
        models (dict): Dictionary of model parameters, e.g., {'amdahl': p, 'gustafson': p, 'usl': (sigma, kappa)}
        show_plots (bool): Whether to display the plots or just save them
        
    Returns:
        dict: Paths to saved plot images
    """
    plot_paths = {}
    
    # Smooth resource levels for plotting models
    max_resource = max(resource_levels) * 1.5  # Extend to show projections
    smooth_x = np.linspace(min(resource_levels), max_resource, 100)
    
    # Plot all models on a single graph
    plt.figure(figsize=(12, 8))
    
    # Plot actual data
    plt.plot(resource_levels, actual_speedups, 'bo', label='Actual Speedup', markersize=8)
    
    # Plot ideal linear speedup
    plt.plot(smooth_x, smooth_x / min(resource_levels), 'k--', label='Ideal Linear Speedup')
    
    # Plot Amdahl's Law if parameters are provided
    if models and 'amdahl' in models:
        p = models['amdahl']
        amdahl_speedups = amdahls_law(p, smooth_x)
        plt.plot(smooth_x, amdahl_speedups, 'r-', label=f"Amdahl's Law (p={p:.3f})")
    
    # Plot Gustafson's Law if parameters are provided
    if models and 'gustafson' in models:
        p = models['gustafson']
        gustafson_speedups = gustafsons_law(p, smooth_x)
        plt.plot(smooth_x, gustafson_speedups, 'g-', label=f"Gustafson's Law (p={p:.3f})")
    
    # Plot Universal Scalability Law if parameters are provided
    if models and 'usl' in models:
        sigma, kappa = models['usl']
        usl_speedups = universal_scalability_law(sigma, kappa, smooth_x)
        plt.plot(smooth_x, usl_speedups, 'm-', label=f"USL (σ={sigma:.3f}, κ={kappa:.3f})")
    
    plt.title('Scalability Models Comparison')
    plt.xlabel('Resource Level')
    plt.ylabel('Speedup')
    plt.grid(True)
    plt.legend()
    
    # Add annotations to explain the models
    explanation_text = ""
    if models and 'amdahl' in models:
        explanation_text += f"Amdahl's Law: {models['amdahl']:.1%} of workload is parallelizable\n"
    if models and 'gustafson' in models:
        explanation_text += f"Gustafson's Law: {models['gustafson']:.1%} of workload scales with resources\n"
    if models and 'usl' in models:
        explanation_text += f"USL: Contention={models['usl'][0]:.3f}, Coherency delay={models['usl'][1]:.3f}"
    
    if explanation_text:
        plt.figtext(0.15, 0.02, explanation_text, ha='left', va='bottom', 
                   bbox={'facecolor':'white', 'alpha':0.8, 'pad':5})
    
    # Save and close figure
    models_plot_path = os.path.join(output_dir, 'scalability_models_comparison.png')
    plt.savefig(models_plot_path, dpi=150)
    plot_paths['models_comparison'] = models_plot_path
    
    if not show_plots:
        plt.close()
    
    return plot_paths

def plot_theoretical_projections(resource_levels, actual_speedups, output_dir, theoretical_projections=None, max_projection=16, show_plots=False):
    """
    Plot theoretical projections for scalability models based on limited data points
    
    Args:
        resource_levels (list): List of resource levels used in tests
        actual_speedups (list): List of observed speedups
        output_dir (str): Directory to save plots
        theoretical_projections (dict): Dictionary of theoretical model projections
        max_projection (int): Maximum resource level to project to
        show_plots (bool): Whether to display the plots or just save them
        
    Returns:
        dict: Paths to saved plot images
    """
    plot_paths = {}
    
    # Create extended range for projections
    max_resource = max(max(resource_levels) * 2, max_projection)
    smooth_x = np.linspace(min(resource_levels), max_resource, 100)
    
    # Plot theoretical projections
    plt.figure(figsize=(14, 10))
    
    # Plot actual data points
    plt.plot(resource_levels, actual_speedups, 'bo', label='Measured Speedup', markersize=10)
    
    # Plot ideal linear speedup
    plt.plot(smooth_x, smooth_x / min(resource_levels), 'k--', label='Ideal Linear Speedup')
    
    # Plot Amdahl's Law projection if available
    if theoretical_projections and 'amdahl' in theoretical_projections:
        p = theoretical_projections['amdahl']['parallelizable_fraction']
        amdahl_speedups = amdahls_law(p, smooth_x)
        plt.plot(smooth_x, amdahl_speedups, 'r-', linewidth=2, 
                 label=f"Amdahl's Law Projection (p={p:.3f})")
        
        # Add theoretical limit line
        max_speedup = theoretical_projections['amdahl']['estimated_max_speedup']
        if max_speedup < 1000:  # Only if it's a reasonable value
            plt.axhline(y=max_speedup, color='r', linestyle=':', alpha=0.6,
                       label=f"Amdahl's Theoretical Limit: {max_speedup:.1f}x")
    
    # Plot Gustafson's Law projection if available
    if theoretical_projections and 'gustafson' in theoretical_projections:
        p = theoretical_projections['gustafson']['scalable_fraction']
        gustafson_speedups = gustafsons_law(p, smooth_x)
        plt.plot(smooth_x, gustafson_speedups, 'g-', linewidth=2,
                 label=f"Gustafson's Law Projection (p={p:.3f})")
    
    # Plot USL projection with typical values if available
    if theoretical_projections and 'usl' in theoretical_projections:
        sigma = theoretical_projections['usl']['typical_sigma']
        kappa = theoretical_projections['usl']['typical_kappa']
        usl_speedups = universal_scalability_law(sigma, kappa, smooth_x)
        plt.plot(smooth_x, usl_speedups, 'm-', linewidth=2,
                 label=f"USL Projection (σ={sigma:.3f}, κ={kappa:.3f})")
        
        # Find peak point for USL
        if kappa > 0:
            peak_n = (1 - sigma) / (2 * kappa) if (1 - sigma) > 0 else 1
            if peak_n <= max_resource:
                peak_speedup = universal_scalability_law(sigma, kappa, peak_n)
                plt.plot([peak_n], [peak_speedup], 'mo', markersize=10)
                plt.annotate(f"USL Peak: {peak_speedup:.1f}x at {peak_n:.1f} resources",
                            xy=(peak_n, peak_speedup), xytext=(peak_n-1, peak_speedup*1.1),
                            arrowprops=dict(facecolor='magenta', shrink=0.05))
    
    # Enhance the plot
    plt.title('Theoretical Scalability Projections', fontsize=16)
    plt.xlabel('Resource Level', fontsize=14)
    plt.ylabel('Speedup', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='best')
    
    # Add explanatory text about the projections
    if theoretical_projections:
        explanation = ["Theoretical Projections Based on Limited Data:"]
        if 'amdahl' in theoretical_projections:
            p = theoretical_projections['amdahl']['parallelizable_fraction']
            max_speedup = theoretical_projections['amdahl']['estimated_max_speedup']
            explanation.append(f"• Amdahl's Law: {p:.1%} parallelizable, max speedup: {max_speedup:.1f}x")
        
        if 'gustafson' in theoretical_projections:
            p = theoretical_projections['gustafson']['scalable_fraction']
            explanation.append(f"• Gustafson's Law: {p:.1%} workload scales with resources")
        
        explanation.append("NOTE: These are projections based on limited data points and should be verified with additional measurements.")
        
        plt.figtext(0.5, 0.01, '\n'.join(explanation), ha='center', va='bottom', 
                   bbox={'facecolor':'lightyellow', 'alpha':0.9, 'pad':10}, fontsize=11)
    
    # Save and close
    projection_path = os.path.join(output_dir, 'theoretical_scalability_projections.png')
    plt.savefig(projection_path, dpi=150, bbox_inches='tight')
    plot_paths['theoretical_projections'] = projection_path
    
    if not show_plots:
        plt.close()
    
    # Generate model comparison plot showing differences between models
    plt.figure(figsize=(12, 9))
    
    # Create a range of resource levels from 1 to max_projection
    x_range = np.arange(1, max_projection+1)
    
    # Plot different models for comparison
    plt.plot(x_range, x_range, 'k-', linewidth=2, label='Ideal Linear')
    
    # Add Amdahl's curves with different p values
    for p in [0.5, 0.75, 0.9, 0.95, 0.99]:
        amdahl_curve = amdahls_law(p, x_range)
        plt.plot(x_range, amdahl_curve, '--', linewidth=1.5, 
                 label=f"Amdahl's (p={p:.2f})")
    
    # Add Gustafson's curves
    for p in [0.5, 0.75, 0.9, 0.95]:
        gustafson_curve = gustafsons_law(p, x_range)
        plt.plot(x_range, gustafson_curve, '-.', linewidth=1.5, 
                 label=f"Gustafson's (p={p:.2f})")
    
    # Add USL curves with different sigma/kappa combinations
    plt.plot(x_range, universal_scalability_law(0.1, 0.01, x_range), 
             ':', linewidth=2, label='USL (σ=0.1, κ=0.01)')
    plt.plot(x_range, universal_scalability_law(0.2, 0.02, x_range), 
             ':', linewidth=2, label='USL (σ=0.2, κ=0.02)')
    
    # Add actual data if available
    if len(resource_levels) > 0:
        plt.plot(resource_levels, actual_speedups, 'ro', markersize=10, label='Your System')
        
        # Add a projection line based on the actual data trend if we have 2+ points
        if len(resource_levels) >= 2:
            z = np.polyfit(resource_levels, actual_speedups, 1)
            p = np.poly1d(z)
            plt.plot(x_range, p(x_range), 'r--', linewidth=2, 
                     label=f"Your Trend: {z[0]:.2f}x + {z[1]:.2f}")
    
    plt.title('Comparative Scalability Models', fontsize=16)
    plt.xlabel('Resource Level', fontsize=14)
    plt.ylabel('Speedup', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='upper left', ncol=2)
    
    # Add explanation
    explanation_text = (
        "This plot shows how different scalability models behave across resource levels.\n"
        "Amdahl's Law shows diminishing returns as resources increase.\n"
        "Gustafson's Law shows more optimistic scaling for problems that grow with resources.\n"
        "USL accounts for both contention and coherency delays, predicting eventual performance decline.\n"
        "The 'p' value represents the proportion of work that can benefit from parallelization."
    )
    
    plt.figtext(0.5, 0.01, explanation_text, ha='center', va='bottom', 
               bbox={'facecolor':'lightyellow', 'alpha':0.9, 'pad':10}, fontsize=11)
    
    # Save and close
    comparison_path = os.path.join(output_dir, 'scalability_model_characteristics.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plot_paths['model_characteristics'] = comparison_path
    
    if not show_plots:
        plt.close()
    
    return plot_paths

def interpret_scalability_results(models):
    """
    Generate human-readable interpretations of the scalability model parameters
    
    Args:
        models (dict): Dictionary of model parameters, e.g., {'amdahl': p, 'gustafson': p, 'usl': (sigma, kappa)}
        
    Returns:
        dict: Interpretations for each model
    """
    interpretations = {}
    
    # Interpret Amdahl's Law results
    if 'amdahl' in models:
        p = models['amdahl']
        interpretations['amdahl'] = {
            'parallelizable_fraction': p,
            'serial_fraction': 1 - p,
            'max_theoretical_speedup': 1 / (1 - p) if p < 1 else float('inf'),
            'assessment': ""
        }
        
        # Add qualitative assessment
        if p > 0.95:
            interpretations['amdahl']['assessment'] = "Excellent parallelizability. The system can scale well with additional resources."
        elif p > 0.8:
            interpretations['amdahl']['assessment'] = "Good parallelizability. The system can benefit significantly from additional resources."
        elif p > 0.5:
            interpretations['amdahl']['assessment'] = "Moderate parallelizability. The system can benefit from additional resources, but with diminishing returns."
        else:
            interpretations['amdahl']['assessment'] = "Poor parallelizability. The system is largely dominated by serial components and will not scale well."
    
    # Interpret Gustafson's Law results
    if 'gustafson' in models:
        p = models['gustafson']
        interpretations['gustafson'] = {
            'scalable_fraction': p,
            'fixed_fraction': 1 - p,
            'assessment': ""
        }
        
        # Add qualitative assessment
        if p > 0.9:
            interpretations['gustafson']['assessment'] = "Excellent scalability with problem size. The system can effectively utilize additional resources as workload grows."
        elif p > 0.7:
            interpretations['gustafson']['assessment'] = "Good scalability with problem size. The system can grow effectively with workload."
        elif p > 0.4:
            interpretations['gustafson']['assessment'] = "Moderate scalability with problem size. The system has some fixed overhead that limits perfect scaling."
        else:
            interpretations['gustafson']['assessment'] = "Poor scalability with problem size. The system has significant fixed overhead that limits scaling with workload growth."
    
    # Interpret Universal Scalability Law results
    if 'usl' in models:
        sigma, kappa = models['usl']
        interpretations['usl'] = {
            'contention_factor': sigma,
            'coherency_delay': kappa,
            'assessment': ""
        }
        
        # Calculate peak point (where adding more resources starts hurting performance)
        if kappa > 0:
            peak_n = (1 - sigma) / (2 * kappa) if (1 - sigma) > 0 else 1
            interpretations['usl']['peak_concurrency'] = max(1, peak_n)
        else:
            interpretations['usl']['peak_concurrency'] = float('inf')
        
        # Add qualitative assessment based on both factors
        if sigma < 0.1 and kappa < 0.01:
            interpretations['usl']['assessment'] = "Excellent scalability. Minimal contention and coherency delays. The system can scale to many processors/resources."
        elif sigma < 0.3 and kappa < 0.05:
            interpretations['usl']['assessment'] = "Good scalability. Some contention but limited coherency issues. The system can scale reasonably well."
        elif sigma < 0.5 and kappa < 0.1:
            interpretations['usl']['assessment'] = "Moderate scalability. Noticeable contention and some coherency delays. Scaling will be limited."
        else:
            interpretations['usl']['assessment'] = "Poor scalability. Significant contention and/or coherency issues. The system will not scale effectively beyond a small number of resources."
    
    return interpretations

def suggest_optimizations(models, interpretations):
    """
    Generate suggestions for system optimization based on scalability analysis
    
    Args:
        models (dict): Dictionary of model parameters
        interpretations (dict): Dictionary of model interpretations
        
    Returns:
        list: Suggested optimizations
    """
    suggestions = []
    
    # Look at Amdahl's Law parameters for optimization suggestions
    if 'amdahl' in models and 'amdahl' in interpretations:
        p = models['amdahl']
        if p < 0.8:
            suggestions.append("Consider optimizing the serial portions of the system, which account for "
                             f"{(1-p)*100:.1f}% of execution time and limit maximum speedup to {1/(1-p):.1f}x.")
        if p > 0.9:
            suggestions.append("The system shows good parallelizability. Consider adding more resources to improve performance.")
    
    # Look at Universal Scalability Law parameters for specific bottleneck identification
    if 'usl' in models and 'usl' in interpretations:
        sigma, kappa = models['usl']
        peak_n = interpretations['usl'].get('peak_concurrency', float('inf'))
        
        if sigma > 0.2:
            suggestions.append(f"High contention factor (σ={sigma:.3f}). Consider reducing shared resource access or "
                             "implementing more efficient locking/synchronization mechanisms.")
        
        if kappa > 0.01:
            suggestions.append(f"Significant coherency delay (κ={kappa:.3f}). Consider optimizing data locality, "
                             "reducing cross-node communication, or implementing better caching strategies.")
        
        if peak_n < float('inf'):
            suggestions.append(f"The system performance is predicted to peak at {peak_n:.1f} resources "
                             "and decline with additional resources. Consider limiting deployment to this size.")
    
    return suggestions
