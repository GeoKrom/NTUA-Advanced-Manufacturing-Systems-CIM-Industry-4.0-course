import pygad
import numpy as np

def cost_func(solution):
    # Constants used:
    NofPoints = 5
    K = 750
    A = 1400
    Plate_Thickness = 0.005  # We choose the Plate Thickness to be 0.005
    PointX = [0, 0.015, 0.025, 0.035, 0.05]
    PointY = 0.024  # The problem is symmetrical so + or  - 0.004 solutions do not differ
    PointZ = 0  # Not sure if it is 0 or Plate_Thickness
    # Gene's value
    InitT = solution[0]
    P = solution[1]
    ElectrodeVelocity = solution[2]
    # Predict using the NN of Ex 2
    T = []
    R1 = []
    R2 = []
    for i in range(len(PointX)):
        # prediction = nn.predict(Plate_Thickness, InitT, P, ElectrodeVelocity, PointX[i], PointY, PointZ)
        prediction = (K+A)/2  # To test the Cost function
        T.append(prediction)
        # Constrain check for each prediction
        if prediction < K:
            R1.append(1)
        else:
            R1.append(0)
        if prediction > A:
            R2.append(1)
        else:
            R2.append(0)

    # Total Cost Calculation
    T = np.power(np.array(T)-(K+A)/2, 2) + np.array(R1)*np.power(np.array(T)-K, 2) + np.array(R2)*np.power(np.array(T)-A, 2)
    cost = np.sum(T)
    return cost


def fitness_func(solution, solution_idx):
    # Fitness function to be used in the GA
    fitness = 1.0 / (cost_func(solution) + 0.000001)
    return fitness


def callback_gen(instance):
    print("Generation : ", instance.generations_completed)
    print("Fitness of the best solution :", instance.best_solution()[1])


# Main Code
# Genetic Algorithm Instance Creation

# Generations and Number of Mating Parents
num_generations = 100
num_parents_mating = 20

# Fitness Function Used
fitness_function = fitness_func

# Population and Chromosomes
sol_per_pop = 50
num_genes = 3
gene_type = [int, int, float]
gene_space = [{'low': 0, 'high': 500},  # For the initTemp parameter
              {'low': 100, 'high': 2000},  # For the P parameter
              {'low': 0.0001, 'high': 1.0}]  # For the ElectrodeVelocity parameter


# Selection Used : Roulette Wheel
parent_selection_type = "rws"
keep_parents = -1  # Choose if you want the parents on the next population
# keep_elitism = 1 to choose the number of best solution to guarantee pass on the next gen

# Crossover Used : Single Point
crossover_type = "single_point"
crossover_probability = None

# Mutation Used
mutation_type = "random"
# mutation_probability=None
mutation_percent_genes = 10

# Instance Creation
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       gene_type=gene_type,
                       gene_space=gene_space,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       crossover_probability=crossover_probability,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes)

ga_instance.run()
callback_gen(ga_instance)
ga_instance.plot_fitness()
ga_instance.plot_genes()
ga_instance.plot_new_solution_rate()
ga_instance.plot_result()