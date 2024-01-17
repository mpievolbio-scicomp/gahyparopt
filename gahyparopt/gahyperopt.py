from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from ipywidgets.widgets.interaction import show_inline_matplotlib_plots
from keras.engine import sequential
from matplotlib import pyplot
from subprocess import Popen, PIPE, STDOUT
from tensorflow.keras import backend as K
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
import glob
import json, io
import numpy as np
import os, shutil, shlex
import owncloud
import pandas
import pandas as pd
import random
import string
import sys
import tensorflow as tf
import time

SHAREURL='https://owncloud.gwdg.de/index.php/s/yKLtY9e230MeuRY'

chars = string.ascii_lowercase
def random_id():
    return ''.join(random.choice(chars) for x in range(4))


class LayerLayout:

    """
    Define a Single Layer Layout
    """
    def __init__(self, layer_type,
                 neurons=None,
                 activation=None,
                 rate=None, ):
        self.neurons = neurons
        self.activation = activation
        self.rate = rate
        self.layer_type = layer_type


    def __call__(self, **kwargs):

        parameters = {
            "neurons": self.neurons,
            "activation": self.activation,
            "rate": self.rate,
            "layer_type": self.layer_type
        }
        parameters.update(kwargs)

        new_layer =  LayerLayout(parameters['layer_type'])

        new_layer.neurons = parameters['neurons']
        new_layer.activation = parameters['activation']
        new_layer.rate = parameters['rate']

        return new_layer


class Chromosome:
    """
    Chromosome Class
    """

    def __init__(self, layer_layout,
                 optimizer,
                 specie,
                 number_of_epochs,
                 steps_per_epoch,
                 parent_a=None,
                 parent_b=None,
                 id=None
                 ):
        self.layer_layout = layer_layout
        self.optimizer = optimizer
        self.number_of_epochs = number_of_epochs
        self.steps_per_epoch = steps_per_epoch
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

    def __call__(self, **kwargs):
        """
        Call method to construct a new instance of this class with
        optionally modified parameters.
        """

        parameters = {
            "layer_layout": self.layer_layout,
            "optimizer": self.optimizer,
            "specie": self.specie,
            "number_of_epochs": self.number_of_epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "parent_a": self.parent_a,
            "parent_b": self.parent_b,
            "id": self.id
        }

        for k,v in kwargs.items():
            if k in parameters.keys():
                parameters[k] = v

        return Chromosome(**parameters)

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
                 number_of_epochs,
                 steps_per_epoch,
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
        self.number_of_epochs = number_of_epochs
        self.steps_per_epoch = steps_per_epoch

    def generate_model_from_chromosome(self, data, chromosome):
        """
        Generate and Train Model using Chromosome Spec
        :param dataframe:
        :param chromosome:
        :return:
        """

        return development(data, chromosome)

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
                number_of_epochs=random.choice(self.number_of_epochs),
                steps_per_epoch=random.choice(self.steps_per_epoch),
                specie=f"I {current}"
            )

            result.append(chromosome)

        run_stop = time.time()
        print(f"Done > Takes {run_stop-run_start} sec")

        return result

    def generate_children(self, mother: Chromosome, father: Chromosome, id=None) -> Chromosome:
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

        # Epochs and Steps
        c_epochs = mother.number_of_epochs if random.randint(0,1) == 0 else father.number_of_epochs
        c_steps = mother.steps_per_epoch if random.randint(0,1) == 0 else father.steps_per_epoch
        c_id = id if id is not None else random_id()

        chromosome = Chromosome(
            layer_layout=c_layer_layout,
            optimizer=c_optimizer,
            specie="",
            number_of_epochs=c_epochs,
            steps_per_epoch=c_steps,
            id=c_id,
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

        # Mutate layers
        for i,layer in enumerate(chromosome.layer_layout):
            if random.random() <= self.layer_mutation_probability:
                chromosome.layer_layout[i] = self.create_random_layer()

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

        # Mutate epochs
        if random.random() <= self.optimizer_mutation_probability:
            chromosome.number_of_epochs = random.choice(self.number_of_epochs)
        if random.random() <= self.optimizer_mutation_probability:
            chromosome.steps_per_epoch = random.choice(self.steps_per_epoch)

        return chromosome


    def evolve_population(self, population, apply_mutations=True):
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

        # Set parents on those new individuals that were copied over.
        for parent in parents:
            new_population.append(
                self.generate_children(
                    mother=parent,
                    father=parent,
                    id=parent.id,
                )
            )

        # Fill Population with new Random Children with Mutation
        while len(new_population) < self.population_size:
            parent_a = random.randint(0, len(parents) - 1)
            parent_b = random.randint(0, len(parents) - 1)
            while parents[parent_a].id == parents[parent_b].id:
                parent_b = random.randint(0, len(parents) - 1)

            if apply_mutations:
                new_population.append(
                    self.mutate_chromosome(
                        self.generate_children(
                            mother=parents[parent_a],
                            father=parents[parent_b]
                        )
                    )
                )
            else:
                new_population.append(
                        self.generate_children(
                            mother=parents[parent_a],
                            father=parents[parent_b]
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


def evaluate_model(ml_model, x, y, model_name="Reference Model", verbose=1):
    """
    Play te Game
    :param ml_model: The model to evaluate.
    :param x: The input (test) data
    :param y: The (test) predictions.
    :return: Performance metrics (loss, accuracy).
    """
    loss, accuracy = ml_model.evaluate(x, y, verbose=verbose)

    return loss, accuracy

def sort_population(population):

    population = sorted(population, key=lambda x: x.accuracy, reverse=True)

    return population

def development(chromosome, data=None):
    # Unpack data.
    x_train, y_train, x_val, y_val = data.values()

    # Fit Model with Data
    history = chromosome.ml_model.fit(
        x_train,
        y_train,
        epochs=chromosome.number_of_epochs,
        steps_per_epoch=chromosome.steps_per_epoch,
        verbose=0,
        validation_data=(x_val, y_val)
    )

    loss, acc = evaluate_model(chromosome.ml_model, x_val, y_val, chromosome.id, 0)
    chromosome.accuracy = acc
    chromosome.loss = loss

    # Write to file if chromosome was read from file.
    if chromosome_file is not None:
        write_chromosome(chromosome_file.split(".json")[0], chromosome)

    return history

def read_chromosome(name):
    with open("{}.json".format(name), 'r') as jso:
        chromosome_dict = json.load(jso)
    return chromosome_from_dict(chromosome_dict)


def timestamp():
    dt = datetime.now()
    ts = dt.strftime("%Y%m%dT%T")
    return ts

def write_chromosome(name, chromosome):
    with open("{}.json".format(name), 'w') as jso:
        json.dump(chromosome, jso, cls=GAJSONEncoder)

class GAJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Chromosome):
            return {"layer_layout": obj.layer_layout,
                    "optimizer": obj.optimizer,
                    "loss": obj.loss,
                    "accuracy": obj.accuracy,
                    "specie": obj.specie,
                    "number_of_epochs": obj.number_of_epochs,
                    "steps_per_epoch": obj.steps_per_epoch,
                    "ml_model": get_model_str(obj.ml_model),
                    "id": obj.id,
                    "parent_a": obj.parent_a,
                    "parent_b": obj.parent_b,
                   }
        if isinstance(obj, LayerLayout):
            return {
                "neurons": obj.neurons,
                "activation": obj.activation,
                "rate": obj.rate,
                "layer_type": obj.layer_type,
            }
        if isinstance(obj, models.Sequential):
            pass
        if isinstance(obj, sequential.Sequential):
            pass
        try:
            return json.JSONEncoder.default(self, obj)
        except:
            raise

def layer_layout_from_dicts(layer_dicts):
    layer_layouts = []
    for dct in layer_dicts:
        layout = LayerLayout(dct['layer_type'])
        layout.neurons = dct['neurons']
        layout.activation = dct['activation']
        layout.rate = dct['rate']

        layer_layouts.append(layout)

    return layer_layouts

def chromosome_from_dict(chromosome_dict):
    layer_layout = layer_layout_from_dicts(chromosome_dict['layer_layout'])
    chromosome = Chromosome(
        layer_layout=layer_layout,
        optimizer=chromosome_dict['optimizer'],
        specie=chromosome_dict['specie'],
        number_of_epochs=chromosome_dict['number_of_epochs'],
        steps_per_epoch=chromosome_dict['steps_per_epoch'],
        id=chromosome_dict['id'],
        parent_a=chromosome_dict['parent_a'],
        parent_b=chromosome_dict['parent_b'],
    )
    chromosome.loss=chromosome_dict['loss']
    chromosome.accuracy=chromosome_dict['accuracy']

    return chromosome


def get_model_str(model):
    model_str = io.StringIO()
    with redirect_stdout(model_str):
        model.summary()
    return model_str.getvalue()


def sync_remote_to_local(name):
    """ Download all json files from the owncloud share. Overwrites all existing files. """

    public_link = SHAREURL

    # Connect to owncloud.
    oc = owncloud.Client.from_public_link(public_link)

    if name == 'all':
        # List content
        fhs = oc.list('.')

        # List of json files.
        jsons = [fh.get_name() for fh in fhs]
        jsons = [fh for fh in jsons if fh.split('.')[-1]=='json']

    else:
        jsons = [name+'.json']

    # Get all json files.
    for j in jsons:
        oc.get_file(j,j)

def delete_local(name=None):
    """ Remove this players json file from the local. Remove all if no `name` given. """

    if name is not None:
        fnames = [name+".json"]
    else:
        fnames =  glob.glob('*.json')

    for fname in fnames:
        os.remove(fname)


def delete_remote(name=None):
    """ Remove this players json file from the remote. Remove all if no `name` given. """
    public_link = SHAREURL

    # Connect to owncloud.
    oc = owncloud.Client.from_public_link(public_link)

    if name is not None:
        fnames = [name+".json"]
    else:
        fnames =  glob.glob('*.json')

    for fname in fnames:
        oc.delete(fname)

def sync_local_to_remote(name):
    """ Upload this players json file to the owncloud share. Overwrites all existing files on the remote side."""

    public_link = SHAREURL

    # Connect to owncloud.
    oc = owncloud.Client.from_public_link(public_link)

    local_file = name+".json" 
    oc.drop_file(local_file)

def git_pull(name):
    command = "scp -oStrictHostKeyChecking=no mplm1023@gwdu20.gwdg.de:/tmp/mplm10/{}.json .".format(name)
    if name == 'all':
        command = "scp -oStrictHostKeyChecking=no mplm1023@gwdu20.gwdg.de:/tmp/mplm10/*.json .".format(name)
    with Popen(shlex.split(command), shell=False, stdout=PIPE, stderr=STDOUT) as proc:
        print(proc.stdout.read())

def git_push(name):
    command = "scp -oStrictHostKeyChecking=no {0:s}.json mplm1023@gwdu20.gwdg.de:/tmp/mplm10/.".format(name)
    with Popen(shlex.split(command), shell=False, stdout=PIPE, stderr=STDOUT) as proc:
        print(proc.stdout.read())

def load_data():
    data = load_mnist()
    return data

def create_start_individuum(ga):
    return ga.generate_first_population_randomly()

def train_individuum(ga,data,individuum):
    clear_keras_session()
    return ga.generate_model_from_chromosome(data, individuum)

def plot_history(history):
    hist_df = pandas.DataFrame(history.history)

    fig, axs = pyplot.subplots(2,1, figsize=(5,5))
    hist_df.plot(y='loss',ax=axs[0], label="Training")
    hist_df.plot(y='val_loss',ax=axs[0], label="Validation")
    axs[0].set_title("Loss")

    hist_df.plot(y='accuracy',ax=axs[1], label="Training")
    hist_df.plot(y='val_accuracy',ax=axs[1], label="Validation")
    axs[1].set_title("Accuracy")

def clear_keras_session():
    # Clear session and reset default graph.
    K.clear_session()
    tf.compat.v1.reset_default_graph()


if __name__ == "__main__":

    chromosome_file = sys.argv[1]
    device = sys.argv[2]

    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

    strategy = tf.distribute.OneDeviceStrategy(device=f'/gpu:0')
    with strategy.scope():

        data = load_mnist()

        chromosome = read_chromosome(chromosome_file.split(".json")[0])

        history = development(
            chromosome,
            data=data
        )

    write_chromosome(chromosome_file.split(".json")[0], chromosome)
