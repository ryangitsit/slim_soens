#%%
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('../')

from neuron import Neuron
from network import Network
import components

from weight_structures import *
from learning_rules import *
from plotting import *
from system_functions import *
from argparser import setup_argument_parser

#%%

def make_rnn(N,p_connect):
    weights = [
        [np.random.rand(3,)]
    ]
    nodes = []
    for n in range(N):
        neuron = Neuron(
            name=f'node_{n}',
            threshold = 0.1,
            weights=weights,
        )
        neuron.normalize_fanin(fanin_factor=2)
        nodes.append(neuron)
    
    for i,n1 in enumerate(nodes):
        for j,n2 in enumerate(nodes):
            if np.random.rand() < p_connect:
                for s,syn in enumerate(n2.synapse_list):
                    if len(syn.incoming) == 0:
                        n1.dend_soma.outgoing.append(syn)
                        syn.incoming.append((n1.dend_soma,1))
                        break
    return nodes
# %%

nodes = make_rnn(10,.5)

def add_clamped_input(nodes,inpt):
    for i,val in enumerate(inpt):
        nodes[i].dend_soma.flux_offset = val*0.5

inpt = np.random.rand(10,)
add_clamped_input(nodes,inpt)

def run_net(nodes,duration=500):
    net = Network(
        run_simulation = True,
        nodes          = nodes,
        duration       = duration,
    )

    plot_nodes(nodes)
run_net(nodes)

# %%

