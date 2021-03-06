import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
import random
import string

chars = string.ascii_lowercase
def random_id():
    return ''.join(random.choice(chars) for x in range(12))
 
    
class LayerLayout:

    """
    Define a Single Layer Layout
    """
    def __init__(self, layer_type):
        self.neurons = None
        self.activation = None
        self.rate = None
        self.layer_type = layer_type

class Chromosome:
    """
    Chromosome Class
    """

    def __init__(self, layer_layout, optimizer, specie, parent_a=None, parent_b=None, id=None):
        self.layer_layout = layer_layout
        self.optimizer = optimizer
#         self.result_worst = None
#         self.result_best = None
#         self.result_avg = None
#         self.result_sum = None
        self.loss = None
        self.accuracy = None
        self.specie = specie
        if id is None:
            self.id = random_id()
        else:
            self.id = id
                
        self.parent_a = parent_a
        self.parent_b = parent_b
        
        # Define Neural Network Topology
        m_model = Sequential()

        # Define Input Layer
        # m_model.add(InputLayer(input_shape=(4,)))
        m_model.add(InputLayer(input_shape=(28*28,))) # corresponding to number of pixels. 

        # Add Hidden Layers
        for layer in self.layer_layout:

            if layer.layer_type == 'dense':
                m_model.add(
                    Dense(
                        layer.neurons,
                        activation=layer.activation
                    )
                )
            elif layer.layer_type == 'dropout':
                m_model.add(
                    Dropout(rate=layer.rate)
                )

        # Define Output Layer
        # m_model.add(Dense(2, activation='sigmoid'))
        m_model.add(Dense(10, activation='softmax'))

        # Compile Neural Network
        m_model.compile(optimizer=self.optimizer,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'],
                       )
        
        self.ml_model = m_model
    
    @property
    def loss(self):
        return self.__loss
    @loss.setter
    def loss(self, val):
        self.__loss = val
        
    @property
    def accuracy(self):
        return self.__accuracy
    @accuracy.setter
    def accuracy(self, val):
        self.__accuracy = val
    
    def __str__(self):
        return str(self.__dict__)
    
    def __eq__(self, other):
        return isinstance(other, Chromosome) and \
               self.id == other.id and \
               self.parent_a == other.parent_a and \
               self.parent_b == other.parent_b
          
               

    def safe_get_hidden_layer_node(self, index=0):
        """
        Return a Hidden Layer Node if Exists, Otherwise, returns None
        :param index:
        :return:
        """

        if len(self.layer_layout) > index:
            return self.layer_layout[index]

        return None

class GADriver(object):
    def __init__(self,
                 layer_counts,
                 no_neurons,
                 rates,
                 activations,
                 layer_types,
                 optimizers,
                 population_size,
                 best_candidates_count,
                 random_candidates_count,
                 optimizer_mutation_probability,
                 layer_mutation_probability,
                 ):
        
        self.layer_counts = layer_counts
        self.no_neurons = no_neurons
        self.rates = rates
        self.activations = activations
        self.layer_types = layer_types
        self.optimizers = optimizers
        self.population_size                  = population_size
        self.best_candidates_count            = best_candidates_count
        self.random_candidates_count          = random_candidates_count
        self.optimizer_mutation_probability   = optimizer_mutation_probability
        self.layer_mutation_probability       = layer_mutation_probability
        
    def generate_model_from_chromosome(self, data, chromosome):
        """
        Generate and Train Model using Chromosome Spec
        :param dataframe:
        :param chromosome:
        :return:
        """
        # Unpack data.
        x_train, y_train, x_val, y_val = data.values()

        
        # Fit Model with Data
        history = chromosome.ml_model.fit(
            x_train,
            y_train,
            epochs=10,
            steps_per_epoch=10,
    #         epochs=chromosome.number_of_epochs,
    #         steps_per_epoch=chromosome.steps_per_epoch,
            verbose=1,
            validation_data=(x_val, y_val)
        )
        
        return history

    def create_random_layer(self):
        """
        Creates a new Randomly Generated Layer
        :return:
        """

        layer_layout = LayerLayout(
            layer_type=self.layer_types[random.randint(0, len(self.layer_types) - 1)]
        )

        if layer_layout.layer_type == 'dense':
            layer_layout.neurons = self.no_neurons[random.randint(0, len(self.no_neurons) - 1)]
            layer_layout.activation = self.activations[random.randint(0, len(self.activations) - 1)]

        elif layer_layout.layer_type == 'dropout':
            layer_layout.rate = self.rates[random.randint(0, len(self.rates) - 1)]

        return layer_layout

    def generate_first_population_randomly(self):
        """
        Creates an Initial Random Population
        :return:
        """

        print("[+] Creating Initial NN Model Population Randomly: ", end='')

        result = []
        run_start = time.time()

        for current in range(self.population_size):

            # Choose Hidden Layer Count
            hidden_layer_counts = self.layer_counts[random.randint(0, len(self.layer_counts)-1)]
            hidden_layer_layout = []

            # Define Layer Structure
            for current_layer in range(hidden_layer_counts):
                hidden_layer_layout.append(self.create_random_layer())

            chromosome = Chromosome(
                layer_layout=hidden_layer_layout,
                optimizer=self.optimizers[random.randint(0, len(self.optimizers)-1)],
                specie=f"I {current}"
            )

            result.append(chromosome)

        run_stop = time.time()
        print(f"Done > Takes {run_stop-run_start} sec")

        return result

    def generate_children(self, mother: Chromosome, father: Chromosome) -> Chromosome:
        """
        Generate a New Children based Mother and Father Genomes
        :param mother: Mother Chromosome
        :param father: Father Chromosome
        :return: A new Children
        """

        # Layer Layout
        c_layer_layout = []
        layers_counts = len(mother.layer_layout) if random.randint(0, 1) == 0 else len(father.layer_layout)
        for ix in range(layers_counts):
            c_layer_layout.append(
                mother.safe_get_hidden_layer_node(ix) if random.randint(0, 1) == 0 else father.safe_get_hidden_layer_node(ix)
            )

        # Remove all Nones on Layers Layout
        c_layer_layout = [item for item in c_layer_layout if item is not None]

        # Optimizer
        c_optimizer = mother.optimizer if random.randint(0, 1) == 0 else father.optimizer

        chromosome = Chromosome(
            layer_layout=c_layer_layout,
            optimizer=c_optimizer,
            specie="",
            id=random_id(),
            parent_a=mother.id,
            parent_b=father.id,
        )
        

        return chromosome


    def mutate_chromosome(self, chromosome):
        """
        Apply Random Mutations on Chromosome
        :param chromosome: input Chromosome
        :return: Result Chromosome. May or May Not Contains a Mutation
        """

        # Apply Mutation on Optimizer
        if random.random() <= self.optimizer_mutation_probability:
            chromosome.optimizer = self.optimizers[random.randint(0, len(self.optimizers)-1)]

        # Apply Mutation on Hidden Layer Size
        if random.random() <= self.layer_mutation_probability:

            new_hl_size = self.layer_counts[random.randint(0, len(self.layer_counts)-1)]

            # Check if Need to Expand or Reduce Layer Count
            if new_hl_size > len(chromosome.layer_layout):

                # Increase Layer Count
                while len(chromosome.layer_layout) < new_hl_size:
                    chromosome.layer_layout.append(
                        self.create_random_layer()
                    )

            elif new_hl_size < len(chromosome.layer_layout):

                # Reduce Layers Count
                chromosome.layer_layout = chromosome.layer_layout[0: new_hl_size]

            else:
                pass  # Do not Change Layer Size

        return chromosome


    def evolve_population(self, population):
        """
        Evolve and Create the Next Generation of Individuals
        :param population: Current Population
        :return: A new population
        """

        print("Evolution ... ")
        # Clear Graphs from Keras e TensorFlow
        K.clear_session()
        tf.compat.v1.reset_default_graph()

        # Select N Best Candidates + Y Random Candidates. Kill the Rest of Chromosomes
        parents = []
        parents.extend(population[0:self.best_candidates_count])  # N Best Candidates
        
        print("*** Old generation ***")
        for p in population:
            print(p.id, p.parent_a, p.parent_b)
        print("*** Parents taken over ***")
        for p in parents:
            print(p.id, p.parent_a, p.parent_b)
            
        for rn in range(self.random_candidates_count):
            parents.append(population[random.randint(self.best_candidates_count, self.population_size - 1)])  # Y Random Candidate
        
        print("*** Random parents ***")
        for p in parents[self.best_candidates_count:]:
            print(p.id, p.parent_a, p.parent_b)
        
        # Create New Population Through Crossover
        new_population = []

        # Fill Population with new Random Children with Mutation
        # Set parents on those new individuals that were copied over.
        for parent in parents:
            new_population.append(
                self.generate_children(
                    mother=parent,
                    father=parent
                )
            )
            
        while len(new_population) < self.population_size:
            parent_a = random.randint(0, len(parents) - 1)
            parent_b = random.randint(0, len(parents) - 1)
            while parents[parent_a].id == parents[parent_b].id:
                parent_b = random.randint(0, len(parents) - 1)
            
            new_population.append(
                self.mutate_chromosome(
                    self.generate_children(
                        mother=parents[parent_a],
                        father=parents[parent_b]
                    )
                )
            )
        
        print("*** New generation ***")
        for p in new_population:
            print(p.id, p.parent_a, p.parent_b)
        
        # Remove parents if already in previous generation
#         for i,p in enumerate(new_population):
#             if p.parent_a is None and p.parent_b is None:
#                 continue
#             for pp in parents:
#                 if p == pp:
                    
#                     print("WARNING:")
#                     print("Removing parents from {}".format(p.id))
#                     print("p:", p.id, p.parent_a, p.parent_b)
#                     print("pp:", pp.id, pp.parent_a, pp.parent_b)
#                     p.parent_a = None
#                     p.parent_b = None
                    
#                     new_population[i] = p
                    
        
#         print("*** New generation after parent cleanup***")
#         for p in new_population:
#             print(p.id, p.parent_a, p.parent_b)
        return new_population

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((60000, 28 * 28))
    x_train = x_train.astype('float32') / 255

    x_test = x_test.reshape((10000, 28 * 28))
    x_test = x_test.astype('float32') / 255

    x_val = x_train[40000:,:]
    x_train = x_train[:40000, :]

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    y_val = y_train[40000:,:]
    y_train = y_train[:40000, :]


    data = {'x_train': x_train,
            'y_train': y_train,
            'x_val': x_val,
            'y_val': y_val}
    
    return data


def generate_reference_ml(data):
    """
    Train and Generate NN Model based on https://github.com/fchollet/deep-learning-with-python-notebooks/blobs/master/2.1-a-first-look-at-a-neural-netword.ipynb'
    :param df: Dataframe to Training Process
    :return:
    """
    print("[+] Training Original NN Model: ", end='')
    run_start = time.time()

    # Define Neural model Topology
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(28 * 28,)))
    model.add(Dense(10, activation='softmax'))

    # Compile Neural model
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit Model with Data
    x_train, y_train, x_val, y_val = data.values()
    training = model.fit(x_train, y_train,
        epochs=20,
        batch_size=128,
        steps_per_epoch=300,
        verbose=1,
        validation_data=(x_val, y_val),
    )

    run_stop = time.time()
    print(f"Done > Takes {run_stop-run_start} sec")

    return model, training


def evaluate_model(ml_model, x, y, model_name="Reference Model"):
    """
    Play te Game
    :param ml_model: The model to evaluate.
    :param x: The input (test) data
    :param y: The (test) predictions.
    :return: Performance metrics (loss, accuracy).
    """
    # TODO: Implement me
    loss, accuracy = ml_model.evaluate(x, y, verbose=1)

    return loss, accuracy
