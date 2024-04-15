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




def make_dataset(**config):
    letters_all = make_letters(patterns='all')
    letters = {}
    for i,(k,v) in enumerate(letters_all.items()):
        if i < config["patterns"]:
            letters[k] = v

    # plot_letters(letters)
            
    config["key_list"] = key_list = list(letters.keys())
    config["keys"]     = '  '.join(key_list)

    weight_type = config["weight_type"]
    if (weight_type    == 'hybrid' 
        or weight_type == 'doubled' 
        or weight_type == 'extended_double_dends'):
        inputs = make_repeated_inputs(letters,20,2)
    else:
        inputs = make_inputs(letters,20)

    return config, letters, inputs

def make_nodes(letters,**config):
    weight_type = config["weight_type"]
    np.random.seed(10)

    nodes = []
    for i,(k,v) in enumerate(letters.items()):

        if weight_type == 'random':
            weights = make_rand_weights()
            arbor_params = None
        elif weight_type == 'uniform':
            weights = make_uniform_weights()
            arbor_params = None
        elif weight_type == 'hifan':
            weights = make_hifan_weights()
            arbor_params = None
        elif weight_type == 'symmetric':
            weights = make_symmetric_weights(p_neg=0.33)
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
            config["doubled"]=True
            weights = make_doubled_weights()
            arbor_params = None
        elif weight_type == 'extended_double_dends':
            config["doubled"]=True
            weights = make_extended_doubled_weights()
            arbor_params = None
            
        neuron = Neuron(
            name='node_'+k,
            threshold = 0.25,
            weights=weights,
            arbor_params=arbor_params,
            )

        # if i >= 0: 
            # dims = [2]
            # for w  in weights:
            #     dims.append(count_total_elements(w))
            # print(dims)
            # graph_adjacency(neuron.adjacency,dims)
                
        neuron.normalize_fanin_symmetric(fanin_factor=config["fan_fact"])

        if weight_type != 'crafted' and config["offset_radius"] != 0: 
            print("randomizing flux")
            neuron.randomize_offsets(config["offset_radius"])

        nodes.append(neuron)

        if config["mutual_inh"] != 0:
            mutual_inhibition(nodes,config["mutual_inh"])

    return nodes, config



def learn(nodes,inputs,**config):


    accs=[]
    class_accs      = [[] for _ in range(config["patterns"])]
    class_successes = np.zeros(config["patterns"])


    for run in range(config["runs"]):

        s1 = time.perf_counter()

        if run%config["print_mod"] == 0 and config["printing"]==True: print_run = True
        else: print_run = False

        if print_run==True:
            print(f"Run {run}")
            print(" "*(14+len(str(run))),config["keys"])

        # np.random.seed(7)
        shuffled = np.arange(0,config["patterns"],1)
        np.random.shuffle(shuffled)
        success = 0
        seen = 0
        for i in shuffled:
            letter = config["key_list"][i]

            targets = np.zeros(config["patterns"])
            targets[i] = config["target"]

            for node in nodes:
                node.add_indexed_spikes(inputs[letter],doubled=config["doubled"])

            net = Network(
                run_simulation = True,
                nodes          = nodes,
                duration       = config["duration"],
            )

            outputs = []
            for nd in range(config["patterns"]):
                outputs.append(len(nodes[nd].dend_soma.spikes))

            pred_idx = np.argmax(outputs)
            pred = config["key_list"][pred_idx]
            errors = targets - outputs

            if no_ties(i,outputs) == True: 
                success += 1
                class_successes[i]+=1
            class_accs[i].append(class_successes[i]/(run+1))
            seen += 1

            # if i == 0: plot_by_layer(nodes[0],4)
            for n,node in enumerate(nodes):
                if config["update_type"] == 'arbor':
                    make_update(node,errors[n],config["eta"],config["max_offset"],config["updater"])
                elif config["update_type"] == 'backpath':
                    backpath(node,errors[n],config["eta"],config["max_offset"],   config["updater"])

            if print_run==True or success==config["patterns"]: 
                print(f"{run} -- {letter} --> {pred}   {outputs}  --  {errors}")
                if config["plotting"] == True:
                    plot_nodes(nodes,title=f"Pattern {letter}",dendrites=True)
                    plot_synapse_inversions(nodes)

                if success==config["patterns"]:
                    print(f"Converged at run {run}!")
                    plot_nodes(nodes,title=f"Pattern {letter}",dendrites=True)
                    plot_synapse_inversions(nodes)
                    return nodes

            # plot_synapse_inversions(nodes,title=f"Pattern {letter}",pattern_idx=i)
            clear_net(net)

        if success==config["patterns"]:
            print(f"Converget at run {run}!")
            break

        acc = success/seen
        accs.append(acc)
        s2 = time.perf_counter()

        if config["printing"]==True:
            if print_run==True: 
                print(f"Run performance:  {np.round(acc,2)}   Run time = {np.round(s2-s1,2)}")
                # print_attrs(nodes[0].dendrite_list,['name','outgoing'])
                print("\n=============")
        else:
            print(f"Run {run} performance:  {np.round(acc,2)}   Run time = {np.round(s2-s1,2)}",end="\r")

    return nodes


# config = {
#     "patterns"          : 12,
#     "runs"              : 1000,
#     "duration"          : 250,
#     "print_mod"         : 50,
#     "plotting"          : False,
#     "realtimeplt"       : False,
#     "printing"          : True,
#     "plot_trajectories" : True,
#     "print_rolls"       : False,

#     ### for arbor
    # "update_type"       : 'arbor',
    # "eta"               : 0.005,
    # "fan_fact"          : 2,
    # "max_offset"        : .4,
    # "target"            : 2,


#     ### for backpath
#     #"update_type"      : 'backpath',
#     #"eta"              : 0.0005,
#     #"fan_fact"         : 2,
#     #"max_offset"       : .8,
#     #"target"           : 10,

#     "offset_radius"     : 0, #0.15,
#     "mutual_inh"        : -0.75,
#     "doubled"           : False,

#     # "weight_type"     : 'hybrid',
#     # "weight_type"     : 'random',
#     # "weight_type"     : 'crafted',
#     # "weight_type"     : 'uniform',
#     # "weight_type"     : 'doubled',
#     # "weight_type"     : 'symmetric',
#     "weight_type"     :   'double_dends',
#     # "weight_type"       'extended_double_dends',
#     # "weight_type"     : 'hifan',
# }

config = setup_argument_parser().__dict__
# config = {}

# config["patterns"] = 3
# config["runs"] = 1000
# config["duration"] = 250
# config["print_mod"] = 50
# config["plotting"] = False
# config["realtimeplt"] = False
# config["printing"] = True
# config["plot_trajectories"] = True
# config["print_rolls"] = False 

# # # config["update_type"] = 'arbor'
# # # config["eta"] = 0.005
# # # config["fan_fact" ] = 2
# # # config["max_offset"] = .4
# # # config["target"] = 2
# # # config["offset_radius"] = 0
# # # config["mutual_inh"] = -0.75
# # # config["doubled"] = False
# # # config["weight_type"] = "double_dends"
# # # config["weight_type"] ='hybrid'
# # # config["weight_type"] ='random'
# # # config["weight_type"] ='crafted'
# # # config["weight_type"] ='uniform'
# # # config["weight_type"] ='doubled'
# # # config["weight_type"] ='symmetric'
# # # config["weight_type"] ='double_dends'
# # # config["weight_type"] ='extended_double_dends'
# # # config["weight_type"] ='hifan'
# # # config["updater"] = "symmetric"

# # # config["update_type" ] =  'backpath'
# # # config["eta"         ] =  0.0005
# # # config["fan_fact"    ] =  2
# # # config["max_offset"  ] =  .8
# # # config["target"      ] =  10
# # # config["offset_radius"] = 0.15
# # # config["weight_type"]  ='uniform'
# # # config["mutual_inh"] = 0

# config["update_type"] = 'arbor'
# config["eta"        ] = 0.005
# config["fan_fact"   ] = 2
# config["max_offset" ] = .4
# config["target"     ] = 2
# config["weight_type"] ='extended_double_dends'
# config["offset_radius"] = 0
# config["mutual_inh"] = -0.75
# config["updater"] = "symmetric" #"chooser" #



print_dict(config)
np.random.seed(10)
config, letters, inputs = make_dataset(**config)
nodes, config           = make_nodes(letters,**config)
nodes                   = learn(nodes,inputs,**config)

# print_attrs(nodes[0].dendrite_list,['name','incoming'])
# print_attrs(nodes[0].dendrite_list,['name','update'])


#%%


def plot_disynaptic_representations(nodes,shape,extended=False):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


    if extended==False:
        fig,ax = plt.subplots(len(nodes),2,figsize=(4,2*len(nodes)), sharex=True, sharey=True)
        for n,node in enumerate(nodes):
            inh_offsets = []
            ext_offsets = []
            for dend in node.dendrite_list:
                if dend.loc[0]==3 and dend.outgoing[0][1] < 0:
                    inh_offsets.append(dend.flux_offset)
                elif dend.loc[0]==3 and dend.outgoing[0][1] > 0:
                    ext_offsets.append(dend.flux_offset)
            ax[n][0].set_xticks([])
            ax[n][1].set_xticks([])

            ax[n][0].set_yticks([])
            ax[n][1].set_yticks([])

            ax[n][0].imshow(np.array(ext_offsets).reshape(shape),cmap='Greens')
            ax[n][1].imshow(np.array(inh_offsets).reshape(shape),cmap='Reds')
        plt.tight_layout()
        plt.show()

    if extended==True:

        fig,ax = plt.subplots(len(nodes),4,figsize=(4,1*len(nodes)), sharex=True, sharey=True)

        for n,node in enumerate(nodes):
            inh_offsets = []
            ext_offsets = []
            for dend in node.dendrite_list:
                if dend.loc[0]==3 and dend.outgoing[0][1] < 0:
                    inh_offsets.append(dend.flux_offset)
                elif dend.loc[0]==3 and dend.outgoing[0][1] > 0:
                    ext_offsets.append(dend.flux_offset)
            ax[n][0].set_xticks([])
            ax[n][1].set_xticks([])

            ax[n][0].set_yticks([])
            ax[n][1].set_yticks([])

            ax[n][0].imshow(np.array(ext_offsets[:9]).reshape(shape),cmap='Greens')
            ax[n][2].imshow(np.array(ext_offsets[9:]).reshape(shape),cmap='Greens')
            ax[n][1].imshow(np.array(inh_offsets[:9]).reshape(shape),cmap='Reds')
            ax[n][3].imshow(np.array(inh_offsets[9:]).reshape(shape),cmap='Reds')

        plt.tight_layout()
        plt.show()

plot_disynaptic_representations(nodes,(3,3),extended=True)
#%%
def plot_trajectories(nodes,double_dends=False):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig,ax = plt.subplots(len(nodes),1,figsize=(8,2.25*len(nodes)), sharex=True)
    for n,node in enumerate(nodes):

        ax[n].set_title(f"Update Trajectory of {node.name} Arbor",fontsize=12)
        for dend in node.dendrite_list:
            if hasattr(dend,'update_traj') and 'ref' not in dend.name:
                if isinstance(dend,components.Soma): 
                    lw = 4
                    c = colors[2]
                    line='solid'

                elif dend.loc[0]==1:
                    c = colors[3] 
                    lw = 2 
                    line = 'dashed'
                elif dend.loc[0]==3:
                    # print(dend.name, dend.outgoing[0][1])
                    if dend.outgoing[0][1] < 0:
                        c = 'r' 
                    else:
                        c = 'g'
                    lw = 2   
                    line = 'dotted'
                else:
                    c = colors[4]
                    lw = 1
                    line = 'dotted'

                ax[n].plot(np.array(dend.update_traj),linewidth=lw,linestyle=line,color=c,label=dend.name)

        # plt.legend(bbox_to_anchor=(1.01,1))
        # ax[n].set_x_label("Updates",fontsize=14)
        # ax[n].set_y_label("Flux Offset",fontsize=14)
    fig.tight_layout()
    plt.show()

# plot_synapse_inversions(nodes)

if config["print_rolls"] == True:
    for node in nodes:
        print_attrs(node.dendrite_list,["name","high_roll","low_roll"])

if config["plot_trajectories"] == True:
    plot_trajectories(nodes,config["doubled"])


