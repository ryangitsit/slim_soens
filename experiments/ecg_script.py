
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


def make_pattern_nodes(patterns,classes,ff=2):
    W = [
    [np.ones((2,))],
    np.ones((2,2)),
    np.ones((4,2)),
    np.ones((8,2)),
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

    if plotting==True:
        plot_nodes(nodes)

    clear_net(net)
    return outputs

# pattern_nodes,chunks = make_pattern_nodes(3,ff=1.25)
# print(len(pattern_nodes))

train = 1600
runs = 1

eta = 0.005
max_offset = 0.1675
updater = 'classic'
duration = 140
patterns = 3
window = int(duration/patterns)
bins = 16
classes = 2

# pattern_nodes,chunks = make_pattern_nodes(patterns,classes,ff=1.25)

def raster_plot_rows(rows):
    times = []
    indices = []
    for i,row in enumerate(rows):
        if len(row)>0:
            for t in row:
                times.append(t)
            indices+=list(np.ones(len(row))*i)
    plt.plot(times,indices,'.k')
    plt.show()


# collect_chunks = [[[] for _ in range(bins)] for _ in range(patterns)]
# collect_chunks_anoms = [[[] for _ in range(bins)] for _ in range(patterns)]

# for run in range(runs):   
#     eta_decay = eta/(run+1)
#     success = 0 
#     for trn in range(train):
#         # raster_plot_rows(reg_spikes[trn])

#         for i, chunk in enumerate(chunks):

#             start = i*window
#             stop = i*window+window
        
#             ### Regular ECG Signals ###
#             in_spikes = reg_spikes[trn]
#             chunk_spikes = spikes_to_chunks(in_spikes,start,stop,bins)
#             for r,row in enumerate(chunk_spikes):
#                 collect_chunks[i][r].append(len(row))
#             # raster_plot_rows(chunk_spikes)
#             targets = np.zeros((patterns*classes,))
#             targets[patterns*(classes-1)+i] = 1

#             outputs = run_net(
#                 pattern_nodes,
#                 chunk_spikes,
#                 targets=targets,
#                 eta=eta_decay,
#                 max_offset=max_offset,
#                 updater=updater,
#                 duration=window,
#                 # plotting=True
#                 )


#             ### Anomolous ECG Signals ###
#             in_spikes = anom_spikes[trn]
#             chunk_spikes = spikes_to_chunks(in_spikes,start,stop,bins)
#             for r,row in enumerate(chunk_spikes):
#                 collect_chunks_anoms[i][r].append(len(row))
#             targets = np.zeros((patterns*classes,))
#             targets[i] = 1

#             outputs = run_net(
#                 pattern_nodes,
#                 chunk_spikes,
#                 targets=targets,
#                 eta=eta_decay,
#                 max_offset=max_offset,
#                 updater=updater,
#                 duration=window
#                 )

#             print(f"Run {run}  --  sample {trn}",end="\r")


# picklit(pattern_nodes,".","ecg_pattern_nodes")
# plot_trajectories(pattern_nodes)

#%%
# mean_chunks=np.flip(np.transpose(np.mean(collect_chunks,axis=2)),axis=1)
# plt.imshow(mean_chunks)

# plt.show()

# mean_chunks_anoms=np.flip(np.transpose(np.mean(collect_chunks_anoms,axis=2)),axis=1)
# plt.imshow(mean_chunks_anoms)
# plt.show()

# chunk_sets = [mean_chunks,mean_chunks_anoms]
# chunk_names = ['A','B','C']
# sig_names = ['Normal','Anomalous']

# fig,ax = plt.subplots(2,3,figsize=(8,5.5), sharex=True,sharey=True)

# for i,chunk_set in enumerate(chunk_sets):
#     for j in range(3):
#         pixels1 = chunk_set[:,j].reshape(4,4)
#         ax[i][j].imshow(pixels1)
#         ax[i][j].set_xticks([])
#         ax[i][j].set_yticks([])
#         if i==1:
#             ax[i][j].set_xlabel(f"{chunk_names[j]}", fontsize=14)
#         if j==0:
#             ax[i][j].set_ylabel(f"{sig_names[i]}", fontsize=14)


# plt.tight_layout()

# fig.text(.1, 1, 'Average Binned Value at Each Input Channel for All Chunks', va='center', fontsize=16)
# # plt.savefig("../results/figs/ecg_pixels")
# plt.show()

# %%

pattern_nodes = picklin(".","ecg_pattern_nodes")

def make_timing_neuron(name=None):
    step = int(141/3)#35
    # W_timing = [
    #     [[.1,.2,.3]]
    # ]

    # arbor_params = [
    #     [[{'tau':(3*step)},{'tau':(2*step)},{'tau':(1*step)}]]
    #     ]

    # W_timing = [
    #     [[.21,.2,.19]]
    # ]

    # arbor_params = [
    #     [[{'tau':(step)},{'tau':(2*step)},{'tau':(3*step)}]]
    #     ]
    

    W_timing = [
        [[.19,.2,.21]]
    ]

    arbor_params = [
        [[{'tau':(3*step)},{'tau':(2*step)},{'tau':(step)}]]
        ]


    # W_timing = [
    #     [[.3,.25,.2]]
    # ]

    # arbor_params = [
    #     [[{'tau':(.5*step)},{'tau':(1*step)},{'tau':(2*step)}]]
    #     ]

    timing_neuron = Neuron(
        name=f'timing_node_{name}',
        threshold = 0.1,
        weights=W_timing,
        arbor_params=arbor_params
        )
    timing_neuron.normalize_fanin_symmetric(fanin_factor=2.5)

    print_attrs(timing_neuron.dendrite_list,['name','tau'])

    return timing_neuron


timing_neuron_reg = make_timing_neuron(name="reg")
timing_neuron_anom = make_timing_neuron(name="anom")

#------------------

# pattern_nodes[3].dend_soma.outgoing.append(timing_neuron_reg.synapse_list[2])
# pattern_nodes[4].dend_soma.outgoing.append(timing_neuron_reg.synapse_list[1])
# pattern_nodes[5].dend_soma.outgoing.append(timing_neuron_reg.synapse_list[0])


# timing_neuron_reg.synapse_list[2].incoming.append((pattern_nodes[3],1))
# timing_neuron_reg.synapse_list[1].incoming.append((pattern_nodes[4],1))
# timing_neuron_reg.synapse_list[0].incoming.append((pattern_nodes[5],1))

# #------------------

# pattern_nodes[0].dend_soma.outgoing.append(timing_neuron_anom.synapse_list[2])
# pattern_nodes[1].dend_soma.outgoing.append(timing_neuron_anom.synapse_list[1])
# pattern_nodes[2].dend_soma.outgoing.append(timing_neuron_anom.synapse_list[0])


# timing_neuron_anom.synapse_list[2].incoming.append((pattern_nodes[0],1))
# timing_neuron_anom.synapse_list[1].incoming.append((pattern_nodes[1],1))
# timing_neuron_anom.synapse_list[0].incoming.append((pattern_nodes[2],1))

# #------------------

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


#------------------

# timing_neuron_anom.dend_soma.outgoing.append(timing_neuron_reg.dend_soma)
# mutual_inhibition([timing_neuron_anom,timing_neuron_reg,],-1)


# for syn in timing_neuron_anom.synapse_list:
#     print(syn.name,syn.incoming)
#     print(syn.incoming,syn.incoming[0][0].name)

# for syn in timing_neuron_reg.synapse_list:
#     print(syn.incoming,syn.incoming[0][0].name)
# for dend in timing_neuron_reg.dendrite_list:
#     if 'ref' not in dend.name: print(dend.name,dend.tau,dend.outgoing,dend.incoming,dend.incoming[0][0].name)





def run_net(
        nodes,inpt,targets=None,
        eta=None,max_offset=None,updater=None,
        duration=150,learn=True,plotting=False,sig="reg"):
    
    for i,neu  in enumerate(nodes):
        if 'timing' not in neu.name:
            neu.add_spike_rows(inpt)
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
        if learn==True:
            error = targets[n] - outputs[n]
            make_update(neuron,error,eta,max_offset,updater)

    if plotting==True:
        # plot_nodes(nodes[:6],dendrites=True)
        # plot_nodes(nodes[6:],dendrites=True,weighting=True)
        # plot_nodes(nodes[-1],dendrites=True,weighting=True)


        plt.figure(figsize=(8,2))
        plt.title("Timing Neuron for ECG Classification",fontsize=14)
        # plt.title("")
        plt.plot(timing_neuron_reg.dend_soma.signal,linewidth=3,label='soma signal')
        # plt.plot(timing_neuron_reg.dend_soma.flux,linewidth=1,label='soma flux')
        branches = ['A','B','C']
        for i,dendrite in enumerate(timing_neuron_reg.dendrite_list[2:]):
            print(dendrite.name, np.max(dendrite.signal), dendrite.tau, dendrite.outgoing[0][1])
            plt.plot(dendrite.flux*dendrite.outgoing[0][1],'--',label=f"Branch {branches[i]} Flux")
            # plt.plot(dendrite.signal,'--',label=dendrite.name)
        plt.ylabel("Signal and Flux",fontsize=14)
        plt.xlabel("Time (ns)",fontsize=14)
        # plt.text(.85, -0.15, 'Anomalous', va='center',fontsize=14)
        if sig=="reg":plt.legend()
        plt.show()
        
        # plot_by_layer(nodes[0],4)

    clear_net(net)

    return outputs,first_spks


nodes = pattern_nodes +[timing_neuron_anom]+[timing_neuron_reg]
test  = 1
print(test)


# train = 1600
# train=np.random.randint(100)
# timing_neuron.dend_soma.threshold=.05
train = 1600
success=0
max_diff = []
centroids = []
linear_acc = 0
line = 90.538
straight_acc = 0
plotting=False

for tst in range(train,train+test):
    # if tst==train:plotting=False #(319+train):plotting=True

    plotting = True
    tst = (319+train)

    outputs_reg,first_spks_reg = run_net(nodes,reg_spikes[tst],learn=False,plotting=plotting,duration=200)
    
    if first_spks_reg[-1] <= line: linear_acc+=1 
    # if outputs_reg[-1] > outputs_reg[-2]: straight_acc+=1
    if first_spks_reg[-1] < first_spks_reg[-2]: straight_acc+=1
    # print(f"  {first_spks_reg[-1],first_spks_reg[-2],first_spks_reg[-1] < first_spks_reg[-2]}")

    outputs_anom,first_spks_anom = run_net(nodes,anom_spikes[tst],learn=False,plotting=plotting,duration=200,sig="anom")
    # print(first_spks_anom,first_spks_reg)
    if first_spks_anom[-1] > line: linear_acc+=1 
    # if outputs_anom[-2] > outputs_anom[-1]: straight_acc+=1
    if first_spks_anom[-1] > first_spks_anom[-2]: straight_acc+=1
    # print(f"  {first_spks_anom[-1],first_spks_anom[-2],first_spks_anom[-1] > first_spks_anom[-2]}")

    # print(first_spks_reg)
    # print(first_spks_anom)

    if outputs_reg[-1]>0 and ((first_spks_reg[-1] < first_spks_anom[-1])):# or outputs_reg[-1] > outputs_anom[-1]):
            hit = 1
    else:

        hit = 0
    diff = first_spks_anom[-1]-first_spks_reg[-1]
    max_diff.append(diff)
    success += hit
    centroids.append(np.min([first_spks_anom[-1],first_spks_reg[-1]])+.5*diff)
    print(tst,outputs_reg,outputs_anom,hit,first_spks_reg[-1],first_spks_anom[-1],diff,end='\r')    


print(f"\nTesting accuracy of {np.round(100*success/(test),2)}")
print(f"\nTesting raw accuracy of {np.round(100*straight_acc/(test*2),2)}\n")
print(f"Max Difference {np.max(max_diff)} at test {np.argmax(max_diff)}")
print(f"Centroid line {np.mean(centroids)}")
print(f"Linear Accuracy {np.round(100*linear_acc/(test*2))}")
# %%

success=0
max_diff = []
centroids = []
linear_acc = 0
for tst in range(train,train+test):
    plotting = False
    outputs_reg,first_spks_reg    = run_net(nodes,reg_spikes[tst],learn=False,plotting=plotting,duration=200)


    outputs_anom,first_spks_anom = run_net(nodes,anom_spikes[tst],learn=False,plotting=plotting,duration=200)



    if outputs_anom[-2]>0 and ((first_spks_reg[-2] > first_spks_anom[-2]) or outputs_reg[-1] < outputs_anom[-1]):
            hit = 1
    else:
        hit = 0
    diff = first_spks_reg[-2]-first_spks_anom[-2]
    max_diff.append(diff)
    success += hit

    print(tst,outputs_reg,outputs_anom,hit,first_spks_reg[-2],first_spks_anom[-2],diff,end='\r')    


print(f"\nTesting accuracy of {np.round(100*success/(test),2)}")
print(f"Max Difference {np.max(max_diff)} at test {np.argmax(max_diff)}")
# print(f"Centroid line {np.mean(centroids)}")   
# print(f"Linear Accuracy {np.round(100*linear_acc/(test*2))}")
# # %%
