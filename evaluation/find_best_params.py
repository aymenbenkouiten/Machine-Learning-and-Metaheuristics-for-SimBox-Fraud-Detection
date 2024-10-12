from hho import HarrisHawksOptimization
import matplotlib.pyplot as plt

def find_best_params(eval, hyperparameter_space, max_iterations, population_size):
    """
    Find the best hyperparameters using Harris Hawks Optimization (HHO).
    
    Parameters:
        eval: Evaluation method/function for the model.
        hyperparameter_space: Dictionary of hyperparameters to search through.
        max_iterations: Maximum number of iterations for optimization.
        population_size: Population size for optimization.
        
    Returns:
        hho: Harris Hawks Optimization object containing optimization results.
    """
    # Initialize Harris Hawks Optimization object
    hho = HarrisHawksOptimization(
        eval,
        dimensions=len(hyperparameter_space),
        pop_size=population_size,
        lb=[hyperparameter_space[param][0] for param in hyperparameter_space],
        ub=[hyperparameter_space[param][1] for param in hyperparameter_space],
        max_iter=max_iterations
    )

    # Perform optimization
    hho.optimize()

    # Plot convergence curve
    plt.plot(range(1, len(hho.convergence_curve) + 1), hho.convergence_curve)
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value')
    plt.title('Convergence Curve')
    plt.grid(True)
    plt.show()

    # Return the HHO object
    return hho