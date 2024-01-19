# Define Hyperparameters for NN
HIDDEN_LAYER_COUNT = [1, 2, 3, ]
HIDDEN_LAYER_NEURONS = [8, 16, 24, 32, 64]
HIDDEN_LAYER_RATE = [0.1, 0.2, 0.3, 0.4]
HIDDEN_LAYER_ACTIVATIONS = ['relu', 'tanh', 'sigmoid']
HIDDEN_LAYER_TYPE = ['dense', 'dropout']
MODEL_OPTIMIZER = ['rmsprop', 'sgd', 'adam']
MODEL_EPOCHS = range(1,11)
MODEL_STEPS_PER_EPOCH = [5, 10, 20, 50, 100]
# Define Genetic Algorithm Parameters
MAX_GENERATIONS = 100  # Max Number of Generations to Apply the Genetic Algorithm
BEST_CANDIDATES_COUNT = 10 # Number of Best Candidates to Use
RANDOM_CANDIDATES_COUNT = 5  # Number of Random Candidates (From Entire Population of Generation) to Next Population
OPTIMIZER_MUTATION_PROBABILITY = 0.1  # 10% of Probability to Apply Mutation on Optimizer Parameter
HIDDEN_LAYER_MUTATION_PROBABILITY = 0.1  # 10% of Probability to Apply Mutation on number of hidden layers.
