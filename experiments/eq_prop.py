#%%
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from keras.datasets import mnist

sys.path.append('../')

from neuron import Neuron
from network import Network
import components

from weight_structures import *
from learning_rules import *
from plotting import *
from system_functions import *
from argparser import setup_argument_parser



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
    
    # for i,n1 in enumerate(nodes):
    #     for j,n2 in enumerate(nodes):
    #         if np.random.rand() < p_connect:
    #             for s,syn in enumerate(n2.synapse_list):
    #                 if len(syn.incoming) == 0:
    #                     n1.dend_soma.outgoing.append(syn)
    #                     syn.incoming.append((n1.dend_soma,1))
    #                     break
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


def make_dataset(digits,samples,start=0):
    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    data = [X_train[(y_train == i)][start:start+samples] for i in range(digits)]

    dataset = [[] for i in range(digits)]

    for i,dig in enumerate(data):
        for j,sample in enumerate(dig):
            dataset[i].append(data[i][j].reshape(784)/255)

    return dataset


def gen_rnn_spikes(N,p,digits,samples,dataset,start,save,exp_name,rnn_nodes=None):
    res_spikes = [[] for _ in range(digits)]

    if rnn_nodes is None:
        nodes = make_rnn(N,p)
        if save==True: 
            picklit(nodes,f"../results/mnist_study/{exp_name}/",f"res_nodes_{N}_{p}_{digits}_{samples}_{start}") 
    else:
        print("Reuse rnn")
        nodes = rnn_nodes
    
    for i in range(digits):
        for j in range(samples):
            print(f"Making dataset: Digit {i} Sample {j}" ,end="\r")
            inpt = dataset[i][j]*10 #> 0

            add_clamped_input(nodes,inpt)
            spikes = run_net(nodes,duration=100)
            res_spikes[i].append(spikes)

    if save==True: picklit(res_spikes,f"../results/mnist_study/{exp_name}/",f"rnn_spikes_{digits}_{samples}_{start}")
    return res_spikes, nodes


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


def learn_readout_mapping(
        digits,
        samples,
        readout_nodes,
        res_spikes,
        runs=1,
        eta=0.0005,
        max_offset=0.4,
        exp_name='test',
        learn=True
        ):
    epoch_accs = []
    for run in range(runs):
        success = 0
        seen = 0
        for i in range(digits):
            for j in range(samples):
                seen+=1
                for node in readout_nodes:

                    node.add_indexed_spikes(res_spikes[i][j],doubled=True)


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
        epoch_accs.append(running_acc)
        print(f"Epoch {run} accuracy = {running_acc}\n")
        if learn==True:
            if run%10==0: picklit(
                readout_nodes,
                f"../results/mnist_study/{exp_name}/",f"readouts_{updater}_{digits}_{samples}_{start}_at_{run}"
                )
            if running_acc == 1:
                print("Converged!")
                picklit(
                    readout_nodes,
                    f"../results/mnist_study/{exp_name}/",f"readouts_converged_{updater}_{digits}_{samples}_{start}_at_{run}"
                    )
                return readout_nodes, epoch_accs
    return readout_nodes, epoch_accs


def get_reservoir_spikes(digits,samples,start,N,p,exp_name,make=False,save=False):

    ### Either Make or Load MNIST Data And Subsequent Reservoir###
    if make == True:
        dataset = make_dataset(digits,samples,start) 
        res_spikes,rnn_nodes = gen_rnn_spikes(N,p,digits,samples,dataset,start,save,exp_name)

        if save==True:
            picklit(dataset,f"../results/mnist_study/{exp_name}/",f"mnist_data_{digits}_{samples}_start_{start}")

    else:
        dataset = picklin(f"../results/mnist_study/{exp_name}/",f"mnist_spikes_{digits}_{samples}_start_{start}")
    return dataset, res_spikes, rnn_nodes
    
    
def train(digits,samples,res_spikes,updater,eta,max_offset,runs,exp_name):
    readout_nodes = make_disynaptic_readout_nodes(digits)
    readout_nodes, learning_accs = learn_readout_mapping(
        digits,
        samples,
        readout_nodes,
        res_spikes,
        runs=runs,
        eta=eta,
        max_offset=max_offset,
        exp_name=exp_name,
        learn=True
        )
    picklit(learning_accs,f"../results/mnist_study/{exp_name}/",f"learning_accs")
    return readout_nodes

def test(digits,samples,start,readout_nodes,rnn_nodes=None):
    test_start = start+digits*samples
    test_samples = int(samples*.2)
    dataset = make_dataset(digits,test_samples,test_start)
    save = True

    test_res_spikes,rnn_nodes = gen_rnn_spikes(
        N,p,digits,test_samples,dataset,test_start,save,exp_name,rnn_nodes=rnn_nodes
        )
    
    readout_nodes, test_accs = learn_readout_mapping(
        digits,
        test_samples,
        readout_nodes,
        test_res_spikes,
        exp_name=exp_name,
        learn=False
        )
    print(f"Testing Accuracy of {test_accs[-1]*100} on {test_samples} new samples of {digits} classes.")
    picklit(test_accs,f"../results/mnist_study/{exp_name}/",f"test_accs")


#%%
    
np.random.seed(10)

digits = 10
samples = 50
start=0
runs = 3


N = 784
p = 1
updater = 'symmetric'
eta = 0.0005
max_offset = 0.4 #0.1675

exp_name = "test_run"

dataset, res_spikes, rnn_nodes = get_reservoir_spikes(digits,samples,start,N,p,exp_name,make=True,save=True)
readout_nodes = train(digits,samples,res_spikes,updater,eta,max_offset,runs,exp_name)
test(digits,samples,start,readout_nodes,rnn_nodes=rnn_nodes)


