import os, shutil, shlex
import json, io
from subprocess import Popen, PIPE, STDOUT
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import tensorflow as tf
import pandas
from matplotlib import pyplot
from ipywidgets.widgets.interaction import show_inline_matplotlib_plots
from keras import backend as K
import sys
from keras.models import Sequential
import owncloud
import glob

SHAREURL='https://owncloud.gwdg.de/index.php/s/yKLtY9e230MeuRY'

from .gahyperopt import GADriver, Chromosome, LayerLayout, evaluate_model, load_mnist

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
        if isinstance(obj, Sequential):
            pass
        if isinstance(obj, tf.python.keras.engine.sequential.Sequential):
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
    breakpoint()
    layer_layout = layer_layout_from_dicts(chromosome_dict['layer_layout'])
    chromosome = Chromosome(
        layer_layout=layer_layout,
        optimizer=chromosome_dict['optimizer'],
        specie=chromosome_dict['specie'],
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

def clear_keras_session():
    K.clear_session()

def train_individuum(ga,data,individuum):
    # Reset tensorflow and keras.
    # clear_keras_session()
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
