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

def run_timing_test(neurons,inpt,duration,plotting=False):
    for i,neu  in enumerate(neurons):
        neu.add_spike_rows(inpt)
    net = Network(
        run_simulation = True,
        nodes          = neurons,
        duration       = duration,
    )
    if plotting == True:
        plot_nodes(neurons)
        for n,neuron in enumerate(neurons):
            plot_by_layer(neuron,3,flux=False)

    outputs = []
    for n,neuron in enumerate(neurons):
        output = len(neurons[n].dend_soma.spikes)
        outputs.append(output)

    clear_net(net)
    return outputs

W_timing = [
#     [[.1,.2,.3,.4]]
# ]

    [[.4,.4,.4,.4]]
]

arbor_params = [
    [[{'tau':(4*35)},{'tau':(3*35)},{'tau':(2*35)},{'tau':(1*35)}]]
    ]

timing_neuron = Neuron(
    name=f'timing_node',
    threshold = .25,
    weights=W_timing,
    arbor_params=arbor_params
    )
timing_neuron.normalize_fanin_symmetric(fanin_factor=2)

downstream_neuron = Neuron(
    name=f'timing_node',
    threshold = .25,
    # weights=W_timing,
    # arbor_params=arbor_params
    )

print_attrs(timing_neuron.dendrite_list,['name','tau','alpha'])

inpt = [
    [10],
    [10],
    [10],
    [10]
]
# inpt = [
#     [10],
#     [50],
#     [90],
#     [130]
# ]

nodes = [timing_neuron,downstream_neuron]
mutual_inhibition(nodes,-1)

timing_neuron.add_spike_rows(inpt)
net = Network(
    run_simulation = True,
    nodes          = nodes,
    duration       = 160,
)
plot_nodes(nodes,dendrites=True,weighting=True)