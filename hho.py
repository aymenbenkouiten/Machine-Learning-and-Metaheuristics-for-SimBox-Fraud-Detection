import numpy as np
import time
import random
import math
from enum import Enum

# Function to generate a Levy flight step
def Levy(dim):
    """
    Generate a Levy flight step.

    Parameters:
        dim (int): Dimensionality of the Levy flight.

    Returns:
        numpy.ndarray: Levy flight step.
    """
    beta=1.5
    sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta) 
    u= 0.01*np.random.randn(dim)*sigma
    v = np.random.randn(dim)
    zz = np.power(np.absolute(v),(1/beta))
    step = np.divide(u,zz)
    return step

# Enumeration for optimization type
class OptimizationType(Enum):
    """
    Enumeration for optimization type (minimization or maximization).
    """
    MINIMIZATION = -1.0
    MAXIMIZATION = 1.0

# Harris Hawks Optimization class
class HarrisHawksOptimization:
    def __init__(self, objective_function, dimensions, pop_size, lb, ub , max_iter=10000, levy=Levy, optimization_type=OptimizationType.MAXIMIZATION, max_fitness=False):
        """
        Constructor method for the HarrisHawksOptimization class.
        
        Parameters:
            objective_function (callable): The objective function to be optimized. 
                It should accept a vector of decision variables as input and return a scalar value.
            dimensions (int): The number of decision variables in the optimization problem.
            pop_size (int): The size of the population or the number of hawks in each iteration.
            lb (float or list/array): Lower bound for each decision variable. 
                If a single value, it is applied to all variables. If a list/array, 
                it specifies the lower bound for each variable individually.
            ub (float or list/array): Upper bound for each decision variable. 
                Similar to lb, it can be a single value or a list/array specifying 
                the upper bound for each variable individually.
            max_iter (int, optional): Maximum number of iterations for the optimization process. 
                Default is 10,000 iterations.
            levy (function, optional): A function that generates Levy flight steps. 
                Default is set to the Levy function provided earlier in the code.
            optimization_type (OptimizationType, optional): An enumeration indicating whether 
                the optimization problem is a maximization or minimization problem. 
                Default is set to OptimizationType.MAXIMIZATION.
            max_fitness (float or False, optional): Maximum fitness value to terminate the optimization 
                process prematurely. If the best fitness value exceeds this threshold, 
                the optimization stops. Default is False, meaning no maximum fitness threshold 
                is enforced.
        """
        self.objective_function = objective_function
        self.max_iter = max_iter
        self.dimensions = dimensions
        self.pop_size = pop_size
        self.levy = levy
        self.optimization_type = optimization_type
        self.max_fitness = max_fitness

        # defining of upper & lower bounds 
        if not isinstance(lb, list):
            self.lower_bound = [lb for _ in range(dimensions)]
            self.upper_bound = [ub for _ in range(dimensions)]
        self.lower_bound = np.asarray(lb)
        self.upper_bound = np.asarray(ub)

        # initial values
        self.current_iter = 0
        self.convergence_curve = np.zeros(max_iter)

        # prey attributes
        self.prey_location = np.zeros(dimensions)
        if optimization_type == OptimizationType.MAXIMIZATION:
            self.prey_energy = float("-inf")
        else:
            self.prey_energy = float("inf")

        # best results
        self.best_solution = None
        if optimization_type == OptimizationType.MAXIMIZATION:
            self.best_fitness = float("-inf")
        else:
            self.best_fitness = float("inf")
    
    # Vérifiez si l'algorithme doit s'arrêter
    def stopping_condition(self):
        """
        Check whether the optimization process should stop based on defined stopping criteria.
        
        Returns:
            bool: True if the optimization should stop, False otherwise.
        """
        if (not self.max_fitness==False) and self.best_fitness > self.max_fitness:
            # Check if a maximum fitness threshold is specified and if the best fitness exceeds it
            return True
        if self.current_iter >= self.max_iter:
            # Check if the maximum number of iterations is reached
            return True
        return False

    # Générer aléatoirement une population initiale de N solutions
    def initialize_population(self):
        """
        Initialize the population of hawks with random positions within the specified bounds.
        
        This method generates a population matrix where each row represents a hawk and each column 
        represents a dimension of the optimization problem. The positions of hawks are randomly 
        initialized within the lower and upper bounds for each dimension.
        """
        self.population = np.random.uniform(0, 1, (self.pop_size, self.dimensions)) *(self.upper_bound-self.lower_bound)+self.lower_bound
        
    def optimize(self):
        """
        Execute the optimization process.

        This method initializes the population, then iteratively runs the optimization until 
        one of the stopping conditions is met. The optimization process involves executing 
        multiple iterations (run_iteration) until the maximum number of iterations is reached 
        or the best fitness exceeds the maximum fitness threshold.
        
        The method measures the execution time of the optimization process.
        """
        startTime = time.time()
        self.initialize_population()
        while not self.stopping_condition():
            self.run_iteration()
        endTime = time.time()
        self.run_time = endTime - startTime
    

    def run_iteration(self):
        """
        Perform one iteration of the Harris Hawks optimization algorithm.

        This method updates the positions of hawks in the population based on the Harris Hawks optimization algorithm.

        Each hawk's position is adjusted considering the exploration and exploitation phases. 
        During the exploration phase, hawks search for new solutions randomly.
        During the exploitation phase, hawks follow the prey and try to improve solutions based on the current best solution.

        The method also updates the best fitness and best solution found so far.
        """
        for i in range(self.pop_size):
            # garder que les faucons qui respect les limites 
            self.population[i, :] = np.clip(self.population[i, :], self.lower_bound, self.upper_bound)

            # Calculate fitness for the current hawk
            fitness = self.objective_function(self.population[i, :])

            # Update the prey's energy and location if necessary
            if(self.optimization_type == OptimizationType.MAXIMIZATION and fitness > self.prey_energy) or (self.optimization_type == OptimizationType.MINIMIZATION and fitness < self.prey_energy):
                self.prey_energy = fitness
                self.prey_location = self.population[i, :].copy()

        # Calculate the escaping energy for exploration phase
        E1 = 2 * (1 - (self.current_iter / self.max_iter))
        
        for i in range(self.pop_size):
            # Randomly generate E0 within range [-1, 1]
            E0 = 2 * random.random() - 1

            # Calculate escaping energy
            escaping_energy = E1 * E0

            if abs(escaping_energy) >= 1: # Phase d'exploration
                # Randomly select a hawk
                q = random.random()
                if q < 0.5:
                    random_hawk = self.population[math.floor(self.pop_size * random.random()), :]
                    r1 = random.random()
                    r2 = random.random()

                    # Update hawk position for exploration
                    self.population[i, :] = random_hawk - r1* abs(random_hawk - 2 * r2 * self.population[i, :])
                else:
                    r3 = random.random()
                    r4 = random.random()

                    # Update hawk position for exploration
                    self.population[i, :] = (self.prey_location - self.population.mean(0)) - r3 * ((self.upper_bound - self.lower_bound) * r4 + self.lower_bound)
            else: # Phase d'exploitation 
                # Generate a random numbers between 0 and 1
                r = random.random()
                r5 = random.random()

                # Calculate the jump strength for the hawk
                jump_strength = 2 * (1 - r5)

                if r >= 0.5:
                    # Exploit by following the prey or jumping towards it
                    if abs(escaping_energy) < 0.5:
                        # Follow the prey directly
                        self.population[i, :] = (self.prey_location) - escaping_energy * abs(self.prey_location - self.population[i, :])
                    else:
                        # Jump towards the prey's location and exploit around it
                        self.population[i, :] = (self.prey_location - self.population[i, :]) - escaping_energy * abs(jump_strength * self.prey_location - self.population[i, :])
                else:
                    # Exploit by exploring around the prey or mean of population
                    if abs(escaping_energy) < 0.5:
                        # Calculate new position around the prey
                        X1 = self.prey_location - escaping_energy * abs(jump_strength * self.prey_location - self.population[i, :] )
                        # Check if the new position improves fitness
                        if self.objective_function(X1) < fitness:
                            self.population[i, :] = X1.copy()
                        else:
                            # If not, perform a Levy flight step and check if it improves fitness
                            X2 = self.prey_location - escaping_energy * abs(jump_strength * self.prey_location - self.population[i, :] ) + np.multiply(np.random.randn(self.dimensions), self.levy(self.dimensions))
                            if self.objective_function(X2) < fitness:
                                self.population[i, :] = X2.copy()
                    else:
                        # Calculate new position around the mean of the population
                        X1 = self.prey_location - escaping_energy * abs(jump_strength * self.prey_location - self.population.mean(0))
                        # Check if the new position improves fitness
                        if self.objective_function(X1) < fitness:
                            self.population[i, :] = X1.copy()
                        else:
                            # If not, perform a Levy flight step and check if it improves fitness
                            X2 = self.prey_location - escaping_energy * abs(jump_strength * self.prey_location - self.population.mean(0)) + np.multiply(np.random.randn(self.dimensions), self.levy(self.dimensions))
                            if self.objective_function(X2) < fitness:
                                self.population[i, :] = X2.copy()

        # Update convergence curve and best solution
        self.convergence_curve[self.current_iter] = self.prey_energy

        # Check if the current prey's energy represents an improvement in fitness
        if(self.optimization_type == OptimizationType.MAXIMIZATION and self.prey_energy > self.best_fitness) or (self.optimization_type == OptimizationType.MINIMIZATION and self.prey_energy < self.best_fitness):
            # Update best fitness and best solution if applicable
            self.best_fitness = self.prey_energy
            self.best_solution = self.prey_location
            print("New best", self.best_fitness)
        # Move to the next iteration
        self.current_iter += 1