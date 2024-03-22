#%%
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

from neuron import Neuron
from network import Network
import components

from plotting import *
from system_functions import *

#%%


def make_rand_weights():
    W = [
    [np.random.rand(3)],
    np.random.rand(3,3)
    ]
    return W

def make_uniform_weights():
    W = [
    [np.ones((3,))],
    np.ones((3,3))
    ]
    return W

def make_doubled_weights():
    W = [
    [np.ones((3,))],
    np.ones((3,3)),
    [[-1,1] for _ in range(9)]
    ]
    arbor_params = [
        [[{'update_type':'normal'} if weight>0 else {'update':'inverted'}
            for w,weight in enumerate(group)] for g,group in enumerate(layer)
        ]
        for l,layer in enumerate(W)]
    return W

def make_symmetric_weights(p_neg=.2):
    W = [
    [[0.3*np.random.choice([-1,1], p=[p_neg,1-p_neg], size=1)[0] for _ in range(3)]],
    [[0.3*np.random.choice([-1,1], p=[p_neg,1-p_neg], size=1)[0] for _ in range(3)] for _ in range(3)]
    ]
    print(W)
    return W

def make_crafted_weights(letter,pixels,symmetry=False):
    count = 0
    synaptice_layer = []
    for i in range(3):
        group = []
        for j in range(3):
            if    symmetry==False:  w=pixels[count]
            elif  pixels[count]==0: w=-1
            else: w=1
            group.append(w)
            count+=1
        synaptice_layer.append(group)

    W = [
    [np.random.rand(3)],
    synaptice_layer
    ]
    return W

def make_double_tree():
    W = [
    [np.ones((3,))],
    np.ones((3,6))
    ]
    print(W)
    return W

def make_hybrid_weights(letter,pixels,symmetry=False):
    count = 0
    synaptice_layer = []
    for i in range(3):
        group = []
        for j in range(3):
            if    symmetry==False:  w=pixels[count]*0.3
            elif  pixels[count]==0: w=-1*0.3
            else: w=0.3
            group.append(w)
            count+=1
        synaptice_layer.append(group)

    W = [
    [np.random.rand(2)],
    np.random.rand(2,3),
    np.concatenate([synaptice_layer,np.random.rand(3,3)])
    ]

    arbor_params = [
        [[{'update':True} if weight!=np.abs(0.3) else {'update':False}
            for w,weight in enumerate(group)] for g,group in enumerate(layer)
        ]
        for l,layer in enumerate(W)]
    return W, arbor_params


def update_offset(dend,update,offmax):
    # print("here")


    if dend.outgoing[0][1] < 0: 
        # print("negative update:",dend.name)
        update*=-1

    dend.flux_offset += update
    if offmax==0: offmax = dend.phi_th
    if dend.flux_offset > 0:
        dend.flux_offset = np.min([dend.flux_offset, offmax])
    elif dend.flux_offset < 0:
        dend.flux_offset = np.max([dend.flux_offset, -1*offmax/2])
    dend.update_traj.append(dend.flux_offset)

def make_update(node,error,eta,offmax):
    for i,dend in enumerate(node.dendrite_list):
        if np.any(dend.flux>0.5):  dend.high_roll += 1
        if np.any(dend.flux<-0.5): dend.low_roll  += 1
        if not hasattr(dend,'update_traj'): dend.update_traj = []
        if (not isinstance(dend,components.Refractory) 
            and not isinstance(dend,components.Soma)):
            if hasattr(dend,'update'):
                if dend.update==True:
                    update = np.mean(dend.signal)*error*eta
                    update_offset(dend,update,offmax)
                    # update_offset(dend,error,eta,offmax)
            else:
                update = np.mean(dend.signal)*error*eta
                update_offset(dend,update,offmax)
                # update_offset(dend,error,eta,offmax)

def backpath(node,error,eta,offmax):
    
    soma = node.dend_soma
    if not hasattr(soma,'update_traj'): soma.update_traj = []

    if np.any(np.mean(soma.signal)>0): ds = 1
    else: ds = 0
    update = ds*error*eta
    
    # update = np.mean(soma.signal)*error*eta
    # if update < 0: update*=0.8
        # update = np.random.rand()*.1 #*np.random.choice([-1,1], p=[.5,.5], size=1)[0]
        # print(soma.name,update)
    # if node.name == 'node_z':print(node.name, error, np.mean(soma.signal), update)
    # update_offset(soma,update,offmax)
    soma.update_traj.append(update)
    
    for dend in node.dendrite_list[2:]:
        if np.any(dend.flux>0.5):  dend.high_roll += 1
        if np.any(dend.flux<-0.5): dend.low_roll  += 1
        if not hasattr(dend,'update_traj'): dend.update_traj = []
        # print(f"{dend.name} -- {dend.outgoing[0][0].name}")

        if np.any(np.mean(dend.signal)>0): ds = 1
        else: ds = 0

        update = ds*dend.outgoing[0][0].update_traj[-1]#*dend.outgoing[0][1] #*eta

        # update = np.mean(dend.signal)*dend.outgoing[0][0].update_traj[-1]*dend.outgoing[0][1]
        # if update < 0: update*=0.8
        # if node.name == 'node_z':
        #     print(
        #         f"{dend.name} -- {dend.outgoing[0][0].name} -- {update} -- {dend.outgoing[0][0].update_traj[-1]} -- {dend.outgoing[0][1]}"
        #         )
        
        update_offset(dend,update,offmax)

            
patterns          = 3

runs              = 200
duration          = 250
print_mod         = 50
plotting          = False
realtimeplt       = False
printing          = True
plot_trajectories = True
print_rolls       = False


## for arbor
# eta        = 0.005
# fan_fact   = 2
# max_offset = .4
# target     = 5

## for backpath
eta        = 0.0005
fan_fact   = 2
max_offset = .8
target     = 10

mutual_inh = 0

# weight_type = 'hybrid'
# weight_type = 'random'
# weight_type = 'crafted'
weight_type = 'uniform'
# weight_type = 'doubled'
# weight_type = 'symmetric'
# weight_type = 'double_dends'
offset_radius = 0.15

update_type = 'backpath'
# update_type = 'arbor'
doubled=False

plt.style.use('seaborn-v0_8-muted')
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


letters_all = make_letters(patterns='all')

letters = {}
for i,(k,v) in enumerate(letters_all.items()):
    if i < patterns:
        letters[k] = v

# plot_letters(letters)
        
inputs = make_inputs(letters,20)
key_list = list(letters.keys())
print(len(set( key_list )),key_list)
keys = '  '.join(key_list)
classes = len(key_list)
print(f"classes = {classes}")


if weight_type == 'hybrid' or weight_type=='doubled':
    inputs = make_repeated_inputs(letters,20,2)
else:
    inputs = make_inputs(letters,20)

np.random.seed(10)
nodes = []
for i,(k,v) in enumerate(letters.items()):

    if weight_type == 'random':
        weights = make_rand_weights()
        arbor_params = None
    elif weight_type == 'uniform':
        weights = make_uniform_weights()
        arbor_params = None
    elif weight_type == 'symmetric':
        weights = make_symmetric_weights(p_neg=0.3)
        arbor_params = None
    elif weight_type == 'crafted':
        weights = make_crafted_weights(k,v,symmetry=True)
        arbor_params = None
    elif weight_type == 'hybrid':
        weights,arbor_params = make_hybrid_weights(k,v)
    elif weight_type == 'doubled':
        weights = make_double_tree()
        arbor_params = None
    elif weight_type == 'double_dends':
        doubled=True
        weights = make_doubled_weights()
        arbor_params = None
        

    neuron = Neuron(
        name='node_'+k,
        threshold = 0.25,
        weights=weights,
        arbor_params=arbor_params,
        )

    if i >= 0: 
        dims = [2]
        for w  in weights:
            dims.append(count_total_elements(w))
        # print(dims)
        # graph_adjacency(neuron.adjacency,dims)
            
    neuron.normalize_fanin_symmetric(fanin_factor=fan_fact)

    if weight_type != 'crafted': neuron.randomize_offsets(offset_radius)

    nodes.append(neuron)


if mutual_inh != 0:
    mutual_inhibition(nodes,mutual_inh)

print_attrs(nodes[0].dendrite_list,['name','incoming'])

# print_attrs(nodes[0].dendrite_list,['name','update'])

#%%
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
        targets[i] = target

        for node in nodes:
            node.add_indexed_spikes(inputs[letter],doubled=doubled)

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
        seen += 1


        for n,node in enumerate(nodes):
            if update_type == 'arbor':
                make_update(node,errors[n],eta,max_offset)
            elif update_type == 'backpath':
                backpath(node,errors[n],eta,max_offset)

        if print_run==True: 
            print(f"{run} -- {letter} --> {pred}   {outputs}  --  {errors}")
            if plotting == True:
                plot_nodes(nodes,title=f"Pattern {letter}",dendrites=True)
        
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
            # plt.legend(bbox_to_anchor=(1.01,1))
            plt.subplots_adjust(right=.85)
        plt.pause(.01)

print("\n")
# print_attrs(nodes[0].dendrite_list,["name","incoming","low_roll"])
if print_rolls == True:
    for node in nodes:
        print_attrs(node.dendrite_list,["name","high_roll","low_roll"])
#%%
# plt.show()

if plot_trajectories == True:
    plt.figure(figsize=(8,4))
    for itr,pattern in enumerate(key_list):
        plt.plot(np.arange(0,len(performance_by_tens[itr]),1),
                np.array(performance_by_tens[itr])+.001*itr,
                color=colors[itr%len(colors)],label=pattern)
    plt.legend()
    plt.show()

    fig,ax = plt.subplots(len(nodes),1,figsize=(8,2.25*len(nodes)), sharex=True)
    for n,node in enumerate(nodes):

        ax[n].set_title(f"Update Trajectory of {node.name} Arbor",fontsize=12)
        for dend in node.dendrite_list:
            if hasattr(dend,'update_traj') and 'ref' not in dend.name:
                if isinstance(dend,components.Soma): 
                    lw = 4
                    c = colors[0]
                    line='solid'
                elif int(dend.name[-5])==1:
                    c = colors[3] 
                    lw = 2 
                    line = 'dashed'
                else:
                    c = colors[1] 
                    lw = 1   
                    line = 'dotted'
                ax[n].plot(np.array(dend.update_traj),linewidth=lw,linestyle=line,label=dend.name)

        # plt.legend(bbox_to_anchor=(1.01,1))
        # ax[n].set_x_label("Updates",fontsize=14)
        # ax[n].set_y_label("Flux Offset",fontsize=14)
    fig.tight_layout()
    plt.show()


    # loc = "results/games/"
    # picklit(accs,loc,f"accs_{fan}_{offmax}")
    # picklit(class_accs,loc,f"classes_{fan}_{offmax}")
    # picklit(nodes,loc,f"nodes_{fan}_{offmax}")
    # del(nodes)