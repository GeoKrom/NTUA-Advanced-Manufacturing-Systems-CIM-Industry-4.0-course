import torch
import pygad.torchga
import pygad
import pandas as pd
import os
import pickle

# TODO: Add docstrings

# *==== Define Constants ====*
WEIGHTS_PATH = "/nn_weights/MaximumTemperatureNN_Model_Parameters.pt"
PREPROCESS_VALUES_PATH = "/nn_weights/MaximumTemperatureNN_training_preprocess_values.pickle"
DATA_PATH = "/data/nn1/"

# *==== Define Parameters for Optimization ====*
# TODO: Parse them using yaml protocol
NUM_SOLUTIONS = 10
NUM_GENERATIONS = 250
NUM_PARENTS_MATING = 5
PARENT_SELECTION_TYPE = "sss"
CROSSOVER_TYPE = "single_point"
MUTATION_TYPE = "random"
MUTATION_PERCENT_GENES = 10
KEEP_PARENTS = -1
SOLUTIONS_PER_POPULATION = 10
NUM_OF_GENES = 20

# *==== Class Declaration ====*


class MaxTemperatureNN(torch.nn.Module):
    """
    Neural Network Model for maximum temperature regression
    of a welding process
    """

    def __init__(self, input=7, output=1):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 32),
            torch.nn.Sigmoid(),
            torch.nn.Linear(32, output)
        )

    def forward(self, x):
        return self.layers(x)

    def predict(self, x):
        with torch.no_grad():
            x = torch.tensor(x)
            prediction = self.forward(x).detach().cpu().numpy()
        return prediction


# *==== Method Definitions ====*

# TODO: Define (again) fitness function
def fitness_func(solution, sol_idx):
    """
    """
    global data_inputs, data_outputs, torch_ga, model, loss_function

    model_weights_dict = pygad.torchga.model_weights_as_dict(model=model,
                                                             weights_vector=solution)

    # Use the current solution as the model parameters.
    model.load_state_dict(model_weights_dict)

    predictions = model(data_inputs)
    abs_error = loss_function(
        predictions, data_outputs).detach().numpy() + 0.00000001

    solution_fitness = 1.0 / abs_error

    return solution_fitness


def generation_callback(ga_instance):
    """
    """
    print("Generation = {generation}".format(
        generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(
        fitness=ga_instance.best_solution()[1]))

    return


def normalize(input, mean, std):
    return (input - mean) / std


def get_data():
    """
    """

    home_folder = os.getcwd()

    # Read and save preporcessing values
    param_filename = home_folder + PREPROCESS_VALUES_PATH
    with open(param_filename, 'rb') as handle:
        train_values_data = pickle.load(handle)
    mean = train_values_data["mean"]
    std = train_values_data["std"]

    train_data_filepath = home_folder + DATA_PATH

    train_df = pd.read_csv(train_data_filepath + "train_dataset.csv")
    X_train, Y_train = train_df.iloc[:, 0:-1], train_df.iloc[:, -1]

    data_input_numpy = normalize(X_train.to_numpy(), mean, std)
    data_input = torch.from_numpy(data_input_numpy).to(torch.float32)
    data_output = torch.from_numpy(Y_train.to_numpy()).to(torch.float32)

    return data_input, data_output


if (__name__ == "__main__"):

    model = MaxTemperatureNN(7, 1)
    loss_function = torch.nn.functional.mse_loss
    data_inputs, data_outputs = get_data()

    # Run genetic algorithm optimization
    torch_ga = pygad.torchga.TorchGA(model=model,
                                     num_solutions=NUM_SOLUTIONS)
    ga_instance = pygad.GA(num_generations=NUM_GENERATIONS,
                           num_parents_mating=NUM_PARENTS_MATING,
                           initial_population=torch_ga.population_weights,
                           fitness_func=fitness_func,
                           parent_selection_type=PARENT_SELECTION_TYPE,
                           crossover_type=CROSSOVER_TYPE,
                           mutation_type=MUTATION_TYPE,
                           mutation_percent_genes=MUTATION_PERCENT_GENES,
                           keep_parents=KEEP_PARENTS,
                           on_generation=generation_callback)

    ga_instance.run()

    # ga_instance.plot_result(
    #     title="Iteration vs. Fitness", linewidth=4)

    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Fitness value of the best solution = {solution_fitness}".format(
        solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(
        solution_idx=solution_idx))

    # Fetch the parameters of the best solution.
    best_solution_weights = pygad.torchga.model_weights_as_dict(model=model,
                                                                weights_vector=solution)
    model.load_state_dict(best_solution_weights)
    predictions = model(data_inputs)
    print("Predictions : n", predictions.detach().numpy())

    abs_error = loss_function(predictions, data_outputs)
    print("Absolute Error : ", abs_error.detach().numpy())
