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

np.random.seed(10)

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



def add_clamped_input(nodes,inpt):
    for i,val in enumerate(inpt):
        nodes[i].dend_soma.flux_offset = val*0.5

def run_net(nodes,duration=500):
    net = Network(
        run_simulation = True,
        nodes          = nodes,
        duration       = duration,
    )
    spikes = net.get_output_spikes()
    clear_net(net)
    return spikes
    # plot_nodes(nodes)


    
# N = 784
# p = .2
# nodes = make_rnn(N,p)

# # inpt = np.random.rand(10,)
# # inpt = np.zeros((10,))
# # inpt[np.random.randint(())] = 1
# inpt = np.random.randint(2, size=N)

# add_clamped_input(nodes,inpt)

# spikes = run_net(nodes,duration=100)

# # %%
# plt.figure(figsize=(10,4))
# plt.plot(spikes[1],spikes[0],'.k',ms=.75)
# plt.show()

#%%

def make_dataset(digits,samples):
    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # X_train = X_train[(y_train == 0) | (y_train == 1) | (y_train == 2)]
    # y = y_train[(y_train == 0) | (y_train == 1) | (y_train == 2)]
    # print(len(X_train))

    data = [X_train[(y_train == i)][:samples] for i in range(digits)]

    dataset = [[] for i in range(digits)]

    for i,dig in enumerate(data):
        for j,sample in enumerate(dig):
            dataset[i].append(data[i][j].reshape(784)/255)

    return dataset

digits = 10
samples = 50
# dataset = make_dataset(digits,samples) 
# picklit(dataset,".","mnist_spikes")
dataset = picklin(".","mnist_spikes")

#%%

N = 784
p = 1

def gen_rnn_spikes(N,p,digits,samples,dataset):
    res_spikes = [[] for _ in range(digits)]

    nodes = make_rnn(N,p)
    for i in range(digits):
        for j in range(samples):
            print(i,j)
            inpt = dataset[i][j]*10 #> 0

            add_clamped_input(nodes,inpt)
            spikes = run_net(nodes,duration=100)
            res_spikes[i].append(spikes)
            # plt.figure(figsize=(10,4))
            # plt.plot(spikes[1],spikes[0],'.k',ms=.75)
            # plt.show()
    picklit(res_spikes,".","rnn_spikes")
    return res_spikes

res_spikes = gen_rnn_spikes(N,p,digits,samples,dataset)
#%%

def make_readout_nodes(classes):
    weights = [
        [np.ones((784,))]
    ]
    readout_nodes = []
    for n in range(classes):
        neuron = Neuron(
            name=f'readout_node_{n}',
            threshold = 0.1,
            weights=weights,
        )
        neuron.normalize_fanin(fanin_factor=2)
        readout_nodes.append(neuron)

    return readout_nodes

def make_disynaptic_readout_nodes(classes):
    weights = [
        [[-1,1] for _ in range(784)]
    ]
    readout_nodes = []
    for n in range(classes):
        neuron = Neuron(
            name=f'readout_node_{n}',
            threshold = 0.1,
            weights=weights,
        )
        neuron.normalize_fanin(fanin_factor=2)
        readout_nodes.append(neuron)

    return readout_nodes


# readout_nodes=make_readout_nodes(digits)
# doubled=False
# updater = 'classic'

readout_nodes=make_disynaptic_readout_nodes(digits)
doubled=True
# updater = 'symmetric'
updater = 'chooser'

learn = True
eta = 0.0005
max_offset = 0.4 #0.1675
runs=2500

for run in range(runs):
    success = 0
    seen = 0
    for i in range(digits):
        for j in range(samples):
            seen+=1
            for node in readout_nodes:
                # print(res_spikes[i][j])

                node.add_indexed_spikes(res_spikes[i][j],doubled=doubled)


            readout_net = Network(
                run_simulation = True,
                nodes          = readout_nodes,
                duration       = 100,
            )
            # spikes = readout_net.get_output_spikes()
            # plot_nodes(readout_nodes)

            targets = np.zeros(digits,)
            targets[i] = 1
            outputs = []
            for n,neuron in enumerate(readout_nodes):
                output = len(readout_nodes[n].dend_soma.spikes)
                outputs.append(output)
                if learn==True:
                    error = targets[n] - outputs[n]
                    make_update(neuron,error,eta,max_offset,updater)
            hit = 0
            if no_ties(i,outputs) == True: 
                hit = 1
            success+=hit
            running_acc=np.round(success/seen,2)
            print(outputs,targets,hit,running_acc)
            clear_net(readout_net)

    print(f"Epoch {run} accuracy = {running_acc}\n")
    if run%10==0: picklit(readout_nodes,".",f"readouts_{updater}_{digits}_{samples}")
    if running_acc == 1:
        print("Converged!")
        picklit(readout_nodes,".",f"readouts_converged_{digits}_{samples}")

#%%
    
plot_trajectories(readout_nodes)

#%%

for node in readout_nodes:
    learned_offsets_positive = []
    learned_offsets_negative = []
    activity_postive = []
    activity_negative = []
    for i,dend in enumerate(node.dendrite_list[2:]):
        if dend.outgoing[0][1] >= 0:
            learned_offsets_positive.append(dend.flux_offset)
            activity_postive.append(np.mean(dend.signal))
        else:
            learned_offsets_negative.append(dend.flux_offset)
            activity_negative.append(np.mean(dend.signal))

    plt.imshow(np.array(learned_offsets_positive).reshape(28,28))
    plt.title("Postive Updates")
    plt.show()
    plt.imshow(np.array(learned_offsets_negative).reshape(28,28))
    plt.title("Negative Updates")
    plt.show()

    plt.imshow(np.array(activity_postive).reshape(28,28))
    plt.title("Postive Activity")
    plt.show()
    plt.imshow(np.array(activity_negative).reshape(28,28))
    plt.title("Negative Activity")
    plt.show()
#%%
    
print(readout_nodes[0].get_dimensions())
print(readout_nodes[0].dendrite_list[10].loc)
print(readout_nodes[0].dend_soma.loc)