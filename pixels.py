#%%
import numpy as np
import matplotlib.pyplot as plt
import time

from neuron import Neuron
from network import Network
import components

from system_functions import *

#%%

letters = make_letters(patterns='all')

del letters['|  ']
del letters['  |']
del letters['_']
del letters['[]']

# plot_letters(letters)
# plot_letters(letters,'v')
inputs = make_inputs(letters,20)

key_list = list(letters.keys())
print(len(set( key_list )))
keys = '  '.join(key_list)
classes = len(key_list)

def make_rand_weights():
    W = [
    [np.random.rand(3)],
    np.random.rand(3,3)
    ]
    return W


def make_update(node,error,eta,offmax=0):
    for i,dend in enumerate(node.dendrite_list):
        if not hasattr(dend,'update_traj'): dend.update_traj = []
        if 'ref' not in dend.name:
            update = np.mean(dend.signal)*error*eta
            dend.flux_offset += update
            if offmax==0: offmax = dend.phi_th
            if dend.flux_offset > 0:
                dend.flux_offset = np.min([dend.flux_offset, offmax])
            elif dend.flux_offset < 0:
                dend.flux_offset = np.max([dend.flux_offset, -1*offmax])
            dend.update_traj.append(dend.flux_offset)
    


runs = 250
duration = 100
eta = 0.00000005

fans = np.arange(0,6,1)
offs = [0,.25,.5]

# for fan in fans:
#     for offmax in offs:


nodes = []
for i,(k,v) in enumerate(letters.items()):
    print(i,k)
    
    neuron = Neuron(
        name='node_'+k,
        threshold = 0.25,
        weights=make_rand_weights(),
        )
    neuron.normalize_fanin(fanin_factor=3)
    nodes.append(neuron)

accs=[]
class_accs = [[] for _ in range(classes)]
class_successes = np.zeros(classes)
for run in range(runs):
    print(" "*15,keys)
    s1 = time.perf_counter()
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

        # neuron = nodes[0]
        # plt.title("signal")
        # for i,dend in enumerate(neuron.dendrite_list):
        #     lw = 2
        #     print(dend.name)
        #     if i==0: lw = 4
        #     plt.plot(dend.signal,linewidth=lw,label=dend.name)
        # plt.legend()
        # plt.show()

        # plt.title("flux")
        # for dend in neuron.dendrite_list:
        #     plt.plot(dend.flux,label=dend.name)
        # plt.legend()
        # plt.show()

        outputs = []
        for nd in range(classes):
            # print(nodes[nd].dend_soma.spikes)
            outputs.append(len(nodes[nd].dend_soma.spikes))

        pred_idx = np.argmax(outputs)
        pred = key_list[pred_idx]
        errors = targets - outputs

        if pred_idx == i: 
            success += 1
            class_successes[i]+=1
            class_accs.append(class_successes[i]/(run+1))

        seen += 1
        for n,node in enumerate(nodes):
            make_update(node,errors[n],eta,0.2)
            for dend in node.dendrite_list:
                dend.signal = []
                dend.flux = []
            for syn in node.synapse_list:
                syn.flux = []

        print(f"{run} -- {letter} --> {pred}   {outputs}  --  {errors}")
        del(net)
    acc = success/seen
    accs.append(acc)
    s2 = time.perf_counter()
    print(f"Run performance:  {np.round(acc,2)}   Run time = {np.round(s2-s1,2)}")

    print("\n=============")


    # loc = "results/games/"
    # picklit(accs,loc,f"accs_{fan}_{offmax}")
    # picklit(class_accs,loc,f"classes_{fan}_{offmax}")
    # picklit(nodes,loc,f"nodes_{fan}_{offmax}")
    # del(nodes)