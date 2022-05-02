import ipywidgets as widgets
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

from gahyparopt.gahyperopt import GADriver, Chromosome, LayerLayout, evaluate_model, load_mnist

from gahyparopt.GAUtilities import *

from gahyparopt.parameters import *


POPULATION_SIZE=1

# init
class game_instance():
    def __init__(self, ga):
        self.data = None
        self.ga = ga
        self.individuum = None
        self.name = None
        self.history = None

spiel = game_instance(
        ga=GADriver(
            layer_counts=HIDDEN_LAYER_COUNT,
            no_neurons=HIDDEN_LAYER_NEURONS,
            rates=HIDDEN_LAYER_RATE,
            activations=HIDDEN_LAYER_ACTIVATIONS,
            layer_types=HIDDEN_LAYER_TYPE,
            optimizers=MODEL_OPTIMIZER,
            population_size=POPULATION_SIZE,
            best_candidates_count=BEST_CANDIDATES_COUNT,
            random_candidates_count=RANDOM_CANDIDATES_COUNT,
            optimizer_mutation_probability=OPTIMIZER_MUTATION_PROBABILITY,
            layer_mutation_probability=HIDDEN_LAYER_MUTATION_PROBABILITY,
        )
    )

widget_dict = {'new_game':None,
               'load_data':None,
               'spieler_name':None,
               'create': None,
               'load': None,
               'train': None,
               'evaluate': None,
               'submit_evaluation':None,
               'log':None,
              }

log_widget = widgets.Output(layout={'border': '1px solid black', 'width':'80%', 'scroll':'true'})
widget_dict['log'] = log_widget

@log_widget.capture()
def log_message(widget, msg, clear=False):
    if clear:
        log_widget.clear_output()
    print("\n {} - {}\n{}\n".format(timestamp(), widget.description, msg))

new_game_button = widgets.Button(description="Neues Spiel")
def new_game_clicked(b):
    log_widget.clear_output()
    spiel.individuum = None
    spiel.data = None
    spiel.name = None
    spiel.history = None
    
    clear_keras_session()
    
    load_data_button.disabled=False
    spieler_name_text.disabled=False
    spieler_name_button.disabled=False
    create_button.disabled=False
    load_button.disabled=True
    train_button.disabled=True
    evaluate_button.disabled=True
    evaluation_submit_button.disabled=True
    
    log_message(b, "Neues Spiel gestartet.", clear=True)
    
    
new_game_button.on_click(new_game_clicked)
widget_dict['new_game'] = new_game_button

load_data_button = widgets.Button(description="Daten laden")
def load_data_clicked(b):
    spiel.data = load_data()
    load_data_button.disabled = True
    
    log_message(b, "MNIST Daten geladen.")
load_data_button.on_click(load_data_clicked)
widget_dict['load_data'] = load_data_button

spieler_name_text = widgets.Text(description="Spieler*in Name:")
spieler_name_button = widgets.Button(description="Spieler*in registrieren")
def spieler_name_clicked(b):
    spiel.name = spieler_name_text.value
    #spieler_name_text.disabled=True
    #spieler_name_button.disabled=True
    
    log_message(b, "{} registriert.".format(spieler_name_text.value))
    
spieler_name_button.on_click(spieler_name_clicked)    

spieler_name_widget = widgets.HBox(children=[spieler_name_text, spieler_name_button])
widget_dict['spieler_name'] = spieler_name_widget

create_button = widgets.Button(description="Gründe neuen Stamm")
def create_button_clicked(b):
    msg = io.StringIO()
    with redirect_stdout(msg):
        spiel.individuum = create_start_individuum(spiel.ga)[0]
        spiel.individuum.ml_model.summary()
   
    log_message(b, msg.getvalue(), True)
    
    create_button.disabled=True
    train_button.disabled=False
    
create_button.on_click(create_button_clicked)
widget_dict['create'] = create_button

load_button = widgets.Button(description="Neue Generation", disabled=True)

def load_button_clicked(b):
    log_message(b, "Lade neue Generation.", clear=True)
    msg = io.StringIO()
    with redirect_stdout(msg):
        clear_keras_session()
        sync_remote_to_local()
        spiel.individuum = read_chromosome(spiel.name)
        spiel.individuum.ml_model.summary()
    log_message(b, msg.getvalue())
    
    load_button.disabled=True
    train_button.disabled=False
    
load_button.on_click(load_button_clicked)
widget_dict['load'] = load_button

train_button = widgets.Button(description = "Individuum entwickeln", disabled=True)
def train_button_clicked(b):
    log_message(b, "Training startet.", True)
    with log_widget:
        try:
            history = train_individuum(spiel.ga,spiel.data,spiel.individuum)
            log_message(b, "Training beendet.", True)
            plot_history(history)
            show_inline_matplotlib_plots()

        except:
            print("WARNING: Training failed, accuracy set to 0, loss set to 100.")
            history = None
    
    
    train_button.disabled=True
    evaluate_button.disabled=False
    
train_button.on_click(train_button_clicked)
    
widget_dict['train'] = train_button

evaluation_value_text = widgets.FloatText(description="Evaluation", disabled=True)
evaluation_submit_button = widgets.Button(description="Jetzt mitteilen", disabled=True)
submit_widget = widgets.HBox(children=[evaluation_value_text, evaluation_submit_button ])
evaluate_button = widgets.Button(description="Individuum evaluieren", disabled=True)

def evaluate_button_clicked(b):
    log_message(b, "Evaluation startet")
    with log_widget:
        try:
            loss, accuracy = evaluate_model(spiel.individuum.ml_model, spiel.data['x_val'], y=spiel.data['y_val'])
        except:
            accuracy = 0.0
            loss = 100.0
    
    evaluate_button.disabled=True
        
    spiel.individuum.loss = loss
    spiel.individuum.accuracy = accuracy
    evaluation_value_text.value = accuracy
    evaluation_submit_button.disabled=False
    
evaluate_button.on_click(evaluate_button_clicked)

widget_dict['evaluate'] = evaluate_button

def evaluation_submit_button_clicked(b):
    log_widget.clear_output()
    with log_widget:
        write_chromosome(spiel.name, spiel.individuum)
        sync_local_to_remote(spiel.name)
    evaluation_submit_button.disabled=True
    load_button.disabled=False
    msg = "Evaluation übermittelt für Spieler*in {}.".format(spiel.name)
    log_message(b, msg)
    
evaluation_submit_button.on_click(evaluation_submit_button_clicked) 

widget_dict['submit_evaluation'] = submit_widget

