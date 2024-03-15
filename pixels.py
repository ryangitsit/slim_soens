#%%
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

from neuron import Neuron
from network import Network
import components
from plotting import plot_nodes

from system_functions import *

#%%

patterns = 5
letters_all = make_letters(patterns='all')

letters = {}
for i,(k,v) in enumerate(letters_all.items()):
    if i < patterns:
        letters[k] = v

# del letters['|  ']
# del letters['  |']
# del letters['_']
# del letters['[]']

# letters = make_letters(patterns='zvn')

# plot_letters(letters)
# plot_letters(letters,'v')
inputs = make_inputs(letters,20)

key_list = list(letters.keys())
print(len(set( key_list )),key_list)
keys = '  '.join(key_list)
classes = len(key_list)
print(f"classes = {classes}")

def make_rand_weights():
    W = [
    [np.random.rand(3)],
    np.random.rand(3,3)
    ]
    return W


def make_update(node,error,eta,offmax):
    for i,dend in enumerate(node.dendrite_list):
        if not hasattr(dend,'update_traj'): dend.update_traj = []
        if 'ref' not in dend.name and 'soma' not in dend.name:
            update = np.mean(dend.signal)*error*eta
            dend.flux_offset += update
            if offmax==0: offmax = dend.phi_th
            if dend.flux_offset > 0:
                dend.flux_offset = np.min([dend.flux_offset, offmax])
            elif dend.flux_offset < 0:
                dend.flux_offset = np.max([dend.flux_offset, -1*offmax/2])
            dend.update_traj.append(dend.flux_offset)
    

runs        = 1000
duration    = 250
print_mod   = 10000
plotting    = False
realtimeplt = False
printing    = False

eta        = 0.0005
fan_fact   = 2
max_offset = 0.4

fans = np.arange(0,6,1)
offs = [0,.25,.5]

plt.style.use('seaborn-muted')
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

np.random.seed(10)
nodes = []
for i,(k,v) in enumerate(letters.items()):
    print(i,k)
    
    neuron = Neuron(
        name='node_'+k,
        threshold = 0.25,
        weights=make_rand_weights(),
        )
    neuron.normalize_fanin(fanin_factor=fan_fact)
    nodes.append(neuron)

accs=[]
class_accs = [[] for _ in range(classes)]
class_successes = np.zeros(classes)
performance_by_tens = [[] for _ in range(classes)] 
for run in range(runs):
    if run%10==0: tenth_samples = np.zeros(classes)
    s1 = time.perf_counter()

    if run%print_mod == 0 and printing==True: print_run = True
    else: print_run = False

    if print_run==True:
        print(f"Run {run}")
        print(" "*(14+len(str(run))),keys)

    shuffled = np.arange(0,classes,1)
    np.random.shuffle(shuffled)
    success = 0
    seen = 0
    for i in shuffled:
        letter = key_list[i]

        targets = np.zeros(classes)
        targets[i] = 5

        for node in nodes:
            node.add_indexed_spikes(inputs[letter])
        
        net = Network(
            run_simulation = True,
            nodes          = nodes,
            duration       = duration,
        )

        outputs = []
        for nd in range(classes):
            outputs.append(len(nodes[nd].dend_soma.spikes))

        pred_idx = np.argmax(outputs)
        pred = key_list[pred_idx]
        errors = targets - outputs

        if no_ties(i,outputs) == True: 
            success += 1
            class_successes[i]+=1
            tenth_samples[i]+=1
        class_accs[i].append(class_successes[i]/(run+1))
        if run%10==0: performance_by_tens[i].append(tenth_samples[i]/10)
        # print(class_accs)
        seen += 1
        for n,node in enumerate(nodes):
            make_update(node,errors[n],eta,max_offset)

        if print_run==True: 
            print(f"{run} -- {letter} --> {pred}   {outputs}  --  {errors}")
            if plotting == True:
                plot_nodes(nodes,title=f"Pattern {letter}")

        clear_net(net)
    if success==classes:
        print(f"Converget at run {run}!")
        break
    acc = success/seen
    accs.append(acc)
    s2 = time.perf_counter()
    if printing==True:
        if print_run==True: 
            print(f"Run performance:  {np.round(acc,2)}   Run time = {np.round(s2-s1,2)}")
            print("\n=============")
    else:
        print(f"Run {run} performance:  {np.round(acc,2)}   Run time = {np.round(s2-s1,2)}",end="\r")


    if realtimeplt==True:
        for itr,pattern in enumerate(key_list):
            if run==0:
                plt.plot(np.arange(0,len(performance_by_tens[itr]),1),
                        performance_by_tens[itr],
                        color=colors[itr%len(colors)],label=pattern)
                # plt.plot(np.arange(0,run+1,1),class_accs[itr],color=colors[itr%len(colors)],label=pattern)
            else:
                plt.plot(np.arange(0,len(performance_by_tens[itr]),1),
                        performance_by_tens[itr],
                        color=colors[itr%len(colors)])
                # plt.plot(np.arange(0,run+1,1),class_accs[itr],color=colors[itr%len(colors)])
            plt.legend(bbox_to_anchor=(1.01,1))
            plt.subplots_adjust(right=.85)
        plt.pause(.01)

plt.show()
for node in nodes:

    plt.figure(figsize=(8,4))
    plt.title(f"Update Trajectory of {node.name} Arbor")
    for dend in node.dendrite_list:
        if hasattr(dend,'update_traj'):
            if isinstance(dend,components.Soma): lw = 4
            else: lw = 2 
            plt.plot(dend.update_traj,linewidth=lw,label=dend.name)
    plt.legend(bbox_to_anchor=(1.01,1))
    plt.tight_layout()
    plt.show()


# loc = "results/games/"
# picklit(accs,loc,f"accs_{fan}_{offmax}")
# picklit(class_accs,loc,f"classes_{fan}_{offmax}")
# picklit(nodes,loc,f"nodes_{fan}_{offmax}")
# del(nodes)