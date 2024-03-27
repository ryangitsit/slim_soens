#%%
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('../')

from neuron import Neuron
from network import Network
import components

from plotting import *
from system_functions import *



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

def make_hifan_weights():
    W = [
    [np.ones((9,))],
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


def make_extended_doubled_weights():
    W = [
    [np.ones((3,))],
    np.ones((3,6)),
    [[-1,1] for _ in range(18)]
    ]

    return W

def make_symmetric_weights(p_neg=.2):
    W = [
    [np.random.rand(3)],
    # [[0.3*np.random.choice([-1,1], p=[p_neg,1-p_neg], size=1)[0] for _ in range(3)]],
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


    # if dend.outgoing[0][1] < 0: 
    #     # print("negative update:",dend.name)
    #     update*=-1

    dend.flux_offset += update
    if offmax==0: offmax = dend.phi_th
    if dend.flux_offset > 0:
        dend.flux_offset = np.min([dend.flux_offset, offmax])
    elif dend.flux_offset < 0:
        dend.flux_offset = np.max([dend.flux_offset, -1*offmax])
    dend.update_traj.append(dend.flux_offset)

def symmetric_udpater(error,eta,dend,offmax):
    """
    Try this for synaptic layer only
    Play with zero-signal update coefficient
    """
    if dend.loc[0]==3:
        if dend.outgoing[0][1] < 0: 
            update_sign = -1
        else:
            update_sign = 1

        if np.mean(dend.signal) > 0:
            update = np.mean(dend.signal)*error*eta*update_sign

        else: 
            update = error*eta*update_sign*-1*.3
    else:
        update = np.mean(dend.signal)*error*eta

    update_offset(dend,update,offmax)

def make_update(node,error,eta,offmax):
    for i,dend in enumerate(node.dendrite_list):
        if np.any(dend.flux>0.5):  dend.high_roll += 1
        if np.any(dend.flux<-0.5): dend.low_roll  += 1
        if not hasattr(dend,'update_traj'): dend.update_traj = []
        if (not isinstance(dend,components.Refractory) 
            and not isinstance(dend,components.Soma)):
            if hasattr(dend,'update'):
                if dend.update==True:
                    symmetric_udpater(error,eta,dend,offmax)

                    # update = np.mean(dend.signal)*error*eta
                    # update_offset(dend,update,offmax)

                    # update_offset(dend,error,eta,offmax)
            else:
                symmetric_udpater(error,eta,dend,offmax)

                # update = np.mean(dend.signal)*error*eta
                # update_offset(dend,update,offmax)

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


# # ## for arbor
# # update_type = 'arbor'
# # eta        = 0.005
# # fan_fact   = 2
# # max_offset = .4
# # target     = 2
# # weight_type = 'double_dends'
# # offset_radius = 0.15
# # mutual_inh = 0 #-0.75
# # doubled=False
            
# patterns          = 10

# runs              = 1000
# duration          = 250
# print_mod         = 50
# plotting          = False
# realtimeplt       = False
# printing          = True
# plot_trajectories = False
# print_rolls       = False


# ## for arbor
# update_type = 'arbor'
# eta        = 0.005
# fan_fact   = 2
# max_offset = .4
# target     = 2



# # ## for backpath
# # update_type = 'backpath'
# # eta        = 0.0005
# # fan_fact   = 2
# # max_offset = .8
# # target     = 10

# offset_radius = 0.15
# mutual_inh = -0.75
# doubled=False

# # weight_type = 'hybrid'
# # weight_type = 'random'
# # weight_type = 'crafted'
# # weight_type = 'uniform'
# # weight_type = 'doubled'
# # weight_type = 'symmetric'
# # weight_type = 'double_dends'
# weight_type = 'extended_double_dends'
# # weight_type = 'hifan'



# plt.style.use('seaborn-v0_8-muted')
# colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

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
        #     dims = [2]
        #     for w  in weights:
        #         dims.append(count_total_elements(w))
        #     print(dims)
        #     graph_adjacency(neuron.adjacency,dims)
                
        neuron.normalize_fanin_symmetric(fanin_factor=config["fan_fact"])

        if weight_type != 'crafted': neuron.randomize_offsets(config["offset_radius"])

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


            for n,node in enumerate(nodes):
                if config["update_type"] == 'arbor':
                    make_update(node,errors[n],config["eta"],config["max_offset"])
                elif config["update_type"] == 'backpath':
                    backpath(node,errors[n],config["eta"],config["max_offset"])

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
                print("\n=============")
        else:
            print(f"Run {run} performance:  {np.round(acc,2)}   Run time = {np.round(s2-s1,2)}",end="\r")

    return nodes


config = {
    "patterns"          : 4,
    "runs"              : 1000,
    "duration"          : 250,
    "print_mod"         : 1,
    "plotting"          : False,
    "realtimeplt"       : False,
    "printing"          : True,
    "plot_trajectories" : False,
    "print_rolls"       : False,

    ### for arbor
    "update_type"       : 'arbor',
    "eta"               : 0.005,
    "fan_fact"          : 2,
    "max_offset"        : .4,
    "target"            : 2,


    ### for backpath
    #"update_type"      : 'backpath',
    #"eta"              : 0.0005,
    #"fan_fact"         : 2,
    #"max_offset"       : .8,
    #"target"           : 10,

    "offset_radius"     : 0.15,
    "mutual_inh"        : -0.75,
    "doubled"           : False,

    # "weight_type"     : 'hybrid',
    # "weight_type"     : 'random',
    # "weight_type"     : 'crafted',
    # "weight_type"     : 'uniform',
    # "weight_type"     : 'doubled',
    # "weight_type"     : 'symmetric',
    "weight_type"     : 'double_dends',
    # "weight_type"       : 'extended_double_dends',
    # "weight_type"     : 'hifan',


}
np.random.seed(10)
config, letters, inputs = make_dataset(**config)
nodes, config           = make_nodes(letters,**config)
nodes                   = learn(nodes,inputs,**config)

# print_attrs(nodes[0].dendrite_list,['name','incoming'])
# print_attrs(nodes[0].dendrite_list,['name','update'])

#%%
node = nodes[7]


# for node in nodes:
#     print(node.name)


# plot_synapse_inversions(nodes)
#%%
print("\n")
# print_attrs(nodes[0].dendrite_list,["name","incoming","low_roll"])
if config["print_rolls"] == True:
    for node in nodes:
        print_attrs(node.dendrite_list,["name","high_roll","low_roll"])
#%%
# plt.show()

if config["plot_trajectories"] == True:
    # plt.figure(figsize=(8,4))
    # for itr,pattern in enumerate(key_list):
    #     plt.plot(np.arange(0,len(performance_by_tens[itr]),1),
    #             np.array(performance_by_tens[itr])+.001*itr,
    #             color=colors[itr%len(colors)],label=pattern)
    # plt.legend()
    # plt.show()

    fig,ax = plt.subplots(len(nodes),1,figsize=(8,2.25*len(nodes)), sharex=True)
    for n,node in enumerate(nodes):

        ax[n].set_title(f"Update Trajectory of {node.name} Arbor",fontsize=12)
        for dend in node.dendrite_list:
            if hasattr(dend,'update_traj') and 'ref' not in dend.name:
                if isinstance(dend,components.Soma): 
                    lw = 4
                    c = colors[0]
                    line='solid'
                elif int(dend.name[dend.name.find("d_")+2])==1:
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