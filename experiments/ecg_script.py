
#%%import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../')

from system_functions import *
from neuron import Neuron
from network import Network
import components

from weight_structures import *
from learning_rules import *
from plotting import *

(reg_spikes, anom_spikes) = picklin(".","ecg_spikes")

idx = np.random.randint(1000)
print(reg_spikes[idx])
print(anom_spikes[idx])


def make_pattern_nodes(patterns):
    W = [
    [np.ones((2,))],
    np.ones((2,2)),
    np.ones((4,2)),
    np.ones((8,2)),
    ]

    chunks = ['A','B','C','D'][:patterns]
    labels = [0,1]

    nodes = []
    for label in labels:
        for chunk in chunks:
            neurons = []
            neuron = Neuron(
                name=f'node_{chunk}{label}',
                threshold = 0.25,
                weights=W,
                )
            neuron.normalize_fanin_symmetric(fanin_factor=2)
            nodes.append(neuron)

    print_attrs(nodes,['name'])
    return nodes,chunks



def spikes_to_chunks(in_spikes,start,stop,bins):
    chunk_spikes = [[] for _ in range(bins)]
    for c,channel in enumerate(in_spikes):
        for spk in channel:
            if spk >= start and spk < stop:
                chunk_spikes[c].append(spk-start)
    return chunk_spikes

#%%

def run_net(
        nodes,inpt,targets=None,
        eta=None,max_offset=None,updater=None,
        duration=150,learn=True,plotting=False):
    for i,neu  in enumerate(nodes):
        if 'timing' not in neu.name:
            neu.add_spike_rows(inpt)
    net = Network(
        run_simulation = True,
        nodes          = nodes,
        duration       = duration,
    )
    outputs = []
    for n,neuron in enumerate(nodes):
        output = len(nodes[n].dend_soma.spikes)
        outputs.append(output)
        if learn==True:
            error = targets[n] - outputs[n]
            make_update(neuron,error,eta,max_offset,updater)

    clear_net(net)
    return outputs


pattern_nodes,chunks = make_pattern_nodes(3)

train = 1000
test  = 1
eta = 0.001
max_offset = 0.8
updater = 'classic'
duration = 140
runs = 3
patterns = 3
window = int(duration/len(chunks))
bins = 16



for run in range(runs):   
    success = 0 
    for trn in range(train):

        for i, chunk in enumerate(chunks):

            start = i*window
            stop = i*window+window
        
            ### Regular ECG Signals ###
            in_spikes = reg_spikes[trn]
            chunk_spikes = spikes_to_chunks(in_spikes,start,stop,bins)

            targets = np.zeros((patterns*2,))
            targets[patterns:] = 1

            outputs = run_net(
                pattern_nodes,
                chunk_spikes,
                targets=targets,
                eta=eta,
                max_offset=max_offset,
                updater=updater,
                duration=window
                )


            ### Anomolous ECG Signals ###
            in_spikes = anom_spikes[trn]
            chunk_spikes = spikes_to_chunks(in_spikes,start,stop,bins)

            targets = np.zeros((patterns*2,))
            targets[:patterns] = 1

            outputs = run_net(
                pattern_nodes,
                chunk_spikes,
                targets=targets,
                eta=eta,
                max_offset=max_offset,
                updater=updater,
                duration=window
                )

            print(f"Run {run}  --  sample {trn}",end="\r")

plot_trajectories(pattern_nodes)
# %%

def make_timing_neuron():
    W_timing = [
        [[.1,.2,.3]]
    ]

    arbor_params = [
        [[{'tau':(3*35)},{'tau':(2*35)},{'tau':(1*35)}]]
        ]

    timing_neuron = Neuron(
        name=f'timing_node',
        threshold = 0.25,
        weights=W_timing,
        arbor_params=arbor_params
        )
    timing_neuron.normalize_fanin_symmetric(fanin_factor=2)

    print_attrs(timing_neuron.dendrite_list,['name','tau'])

    return timing_neuron

#%%
timing_neuron = make_timing_neuron()


pattern_nodes[3].dend_soma.outgoing.append(timing_neuron.synapse_list[0])
pattern_nodes[4].dend_soma.outgoing.append(timing_neuron.synapse_list[1])
pattern_nodes[5].dend_soma.outgoing.append(timing_neuron.synapse_list[2])


timing_neuron.synapse_list[0].incoming.append((pattern_nodes[3],1))
timing_neuron.synapse_list[1].incoming.append((pattern_nodes[4],1))
timing_neuron.synapse_list[2].incoming.append((pattern_nodes[5],1))

#%%

nodes = pattern_nodes +[timing_neuron]
test  = 100
for tst in range(train,train+test):
    outputs_reg = run_net(nodes,reg_spikes[tst],learn=False)
    outputs_anom = run_net(nodes,anom_spikes[tst],learn=False)
    if outputs_reg[-1]>outputs_anom[-1]:
        hit = 1
    else:
        hit = 0
    success += hit
    print(outputs_reg,outputs_anom,hit)    

print(f"Testing accuracy of {np.round(100*success/test,2)}")
# %%
