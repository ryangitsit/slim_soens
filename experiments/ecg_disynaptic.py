
import numpy as np
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

(reg_spikes, anom_spikes) = picklin(".","../results/ecg/ecg_spikes")


def make_pattern_nodes(patterns,classes,ff=2):
    W = [
    [np.ones((2,))],
    np.ones((2,2)),
    np.ones((4,2)),
    np.ones((8,2)),
    [[-1,1] for _ in range(16)]
    ]

    chunks = ['A','B','C','D'][:patterns]
    labels = np.arange(0,classes,1)

    nodes = []
    for label in labels:
        for chunk in chunks:
            neurons = []
            neuron = Neuron(
                name=f'node_{chunk}{label}',
                threshold = 0.1,
                weights=W,
                )
            neuron.normalize_fanin_symmetric(fanin_factor=ff)
            nodes.append(neuron)

    print_attrs(nodes,['name'])
    return nodes,chunks

def spikes_to_chunks(in_spikes,start,stop,bins):
    chunk_spikes = [[] for _ in range(bins)]
    for c,channel in enumerate(in_spikes):
        for spk in channel:
            if spk >= start and spk < stop:
                chunk_spikes[c].append(spk-start) #***
    return chunk_spikes

def run_net(
        nodes,
        inpt,
        targets    = None,
        eta        = None,
        max_offset = None,
        updater    = None,
        duration   = 150,
        learn      = True,
        plotting   = False
        ):
    for i,neu  in enumerate(nodes):
        if 'timing' not in neu.name:
            neu.add_spike_rows_doubled(inpt)
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
            make_update(neuron,error,eta,max_offset,updater,traj=True)

    if plotting==True:
        plot_nodes(nodes)

    clear_net(net)
    return outputs


def learn_chunks(
        pattern_nodes,
        chunks, 
        train      = 100,
        runs       = 1,
        eta        = 0.005,
        max_offset = 0.1675,
        updater    = 'symmetric',
        duration   = 140,
        patterns   = 3,
        bins       = 16,
        classes    = 2,
        plotting   = False,
    ):

    window = int(duration/patterns)
    collect_chunks = [[[] for _ in range(bins)] for _ in range(patterns)]
    collect_chunks_anoms = [[[] for _ in range(bins)] for _ in range(patterns)]
    laccs = []
    for run in range(runs):   
        # eta_decay = eta/(run+1)
        # if run%50==0: plotting=True
        # else:plotting=False
        for trn in range(train):
            # raster_plot_rows(reg_spikes[trn])
            success = 0
            for i, chunk in enumerate(chunks):

                start = i*window
                stop = i*window+window
            
                ### Regular ECG Signals ###
                in_spikes = reg_spikes[trn]
                chunk_spikes = spikes_to_chunks(in_spikes,start,stop,bins)
                for r,row in enumerate(chunk_spikes):
                    collect_chunks[i][r].append(len(row)) #***
                # raster_plot_rows(chunk_spikes)
                targets = np.zeros((patterns*classes,))
                targets[patterns*(classes-1)+i] = 1
                # print(targets)
                outputs = run_net(
                    pattern_nodes,
                    chunk_spikes,
                    targets=targets,
                    eta=eta,
                    max_offset=max_offset,
                    updater=updater,
                    duration=window,
                    plotting=plotting
                    # plotting=True
                    )
               
                if np.argmax(targets[patterns*(classes-1)+i])==np.argmax(outputs) and sum(outputs)>0:
                    hit = 1
                    success+=1
                else: 
                    hit = 0
                # print(targets,outputs,hit)

                ### Anomolous ECG Signals ###
                in_spikes = anom_spikes[trn]
                chunk_spikes = spikes_to_chunks(in_spikes,start,stop,bins)
                for r,row in enumerate(chunk_spikes):
                    collect_chunks_anoms[i][r].append(len(row))
                targets = np.zeros((patterns*classes,))
                targets[i] = 1
                
                outputs = run_net(
                    pattern_nodes,
                    chunk_spikes,
                    targets=targets,
                    eta=eta,
                    max_offset=max_offset,
                    updater=updater,
                    duration=window,
                    plotting=plotting
                    )
                
                if np.argmax(targets[patterns*(classes-1)+i])==np.argmax(outputs) and sum(outputs)>0:
                    hit = 1
                    success+=1
                else: 
                    hit = 0
                # print(targets,outputs,hit)

                
                # print(f"Run {run}  --  sample {trn}",end="\r")
            acc = rounded_percentage(success,(patterns*classes))
            # print("-"*10,acc,"-"*10)
            laccs.append(acc)
    plt.plot(laccs)
    plt.show()
    return pattern_nodes


# pattern_nodes = picklin("../results/ecg/","ecg_pattern_nodes")

def make_timing_neuron(name=None):
    step = int(141/3)#35

    W_timing = [
        [[.19,.2,.21]]
    ]

    arbor_params = [
        [[{'tau':(3*step)},{'tau':(2*step)},{'tau':(step)}]]
        ]

    timing_neuron = Neuron(
        name=f'timing_node_{name}',
        threshold = 0.1,
        weights=W_timing,
        arbor_params=arbor_params
        )
    
    timing_neuron.normalize_fanin_symmetric(fanin_factor=2.5)

    # print_attrs(timing_neuron.dendrite_list,['name','tau'])

    return timing_neuron

def connect_pattern_layer_to_timing_neurons(
        pattern_nodes,
        timing_neuron_reg,
        timing_neuron_anom
        ):
    
    pattern_nodes[3].dend_soma.outgoing.append(timing_neuron_reg.synapse_list[0])
    pattern_nodes[4].dend_soma.outgoing.append(timing_neuron_reg.synapse_list[1])
    pattern_nodes[5].dend_soma.outgoing.append(timing_neuron_reg.synapse_list[2])


    timing_neuron_reg.synapse_list[0].incoming.append((pattern_nodes[3],1))
    timing_neuron_reg.synapse_list[1].incoming.append((pattern_nodes[4],1))
    timing_neuron_reg.synapse_list[2].incoming.append((pattern_nodes[5],1))

    #------------------

    pattern_nodes[0].dend_soma.outgoing.append(timing_neuron_anom.synapse_list[0])
    pattern_nodes[1].dend_soma.outgoing.append(timing_neuron_anom.synapse_list[1])
    pattern_nodes[2].dend_soma.outgoing.append(timing_neuron_anom.synapse_list[2])


    timing_neuron_anom.synapse_list[0].incoming.append((pattern_nodes[0],1))
    timing_neuron_anom.synapse_list[1].incoming.append((pattern_nodes[1],1))
    timing_neuron_anom.synapse_list[2].incoming.append((pattern_nodes[2],1))

def run_classification(
        nodes,
        inpt,
        targets     = None,
        eta         = None,
        max_offset  =  None,
        updater     = None,
        duration    = 150,
        learn       = True,
        plotting    = False,
        sig         = "reg"
        ):
    
    for i,neu  in enumerate(nodes):
        if 'timing' not in neu.name:
            neu.add_spike_rows_doubled(inpt)

    net = Network(
        run_simulation = True,
        nodes          = nodes,
        duration       = duration,
    )

    outputs = []
    first_spks = []
    for n,neuron in enumerate(nodes):
        output = len(nodes[n].dend_soma.spikes)
        outputs.append(output)
        if output > 0:
            first_spks.append(nodes[n].dend_soma.spikes[0])
        else:
            first_spks.append(500)

    clear_net(net)

    return outputs,first_spks


params = {
    "train"      : 1,
    "runs"       : 1000,
    "eta"        : 0.05,
    "max_offset" : 0.4, #0.1675,
    "updater"    : 'symmetric',
    "duration"   : 140,
    "patterns"   : 3,
    "bins"       : 16,
    "classes"    : 2,
}


pattern_nodes,chunks = make_pattern_nodes(params["patterns"],params["classes"],ff=1.25)

learn_chunks(pattern_nodes,chunks,**params)
plot_trajectories(pattern_nodes)

# picklit(pattern_nodes,"../results/ecg/","ecg_pattern_nodes_disyn")


# pattern_nodes = picklin("../results/ecg/","ecg_pattern_nodes_disyn")

# timing_neuron_reg = make_timing_neuron(name="reg")
# timing_neuron_anom = make_timing_neuron(name="anom")

# nodes = pattern_nodes +[timing_neuron_anom]+[timing_neuron_reg]
# test  = 20
# print(f"Testing on {test} unseen samples.")

# reg_hits  = 0
# anom_hits = 0

# # for tst in range(params["train"],params["train"]+test):
# for tst in range(test):
#     outputs_reg,first_spks_reg   = run_classification(nodes,reg_spikes[tst],learn=False,duration=200)
#     outputs_anom,first_spks_anom = run_classification(nodes,anom_spikes[tst],learn=False,duration=200,sig="anom")
#     print(tst,outputs_reg,outputs_anom," -- ",first_spks_reg,first_spks_anom,end='\r')

#     if sum(outputs_reg[:3]) < sum(outputs_reg[3:6]): reg_hits +=1
#     if sum(outputs_anom[:3]) > sum(outputs_anom[3:6]): anom_hits +=1

# print(f"\n\nTotal pattern-comparison accuracy = {rounded_percentage(reg_hits+anom_hits,test*2)}")
# print(f"Reg pattern-comparison accuracy = {rounded_percentage(reg_hits,test)}")
# print(f"Anom pattern-comparison accuracy = {rounded_percentage(anom_hits,test)}")


# print_attrs(nodes,["name"])