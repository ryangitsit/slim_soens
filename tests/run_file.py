import numpy as np
import matplotlib.pyplot as plt
import timeit
import time
import sys
sys.path.append('../slim_soens/slim_soens')
sys.path.append('../')

from neuron import Neuron
from network import Network

from system_functions import *
# from plotting import heatmap_adjacency


def main():

#     weights = [
#         [[0.2,0.5]],
#         [[0.2,0.2],[0.5,0.5]],
#         [[.21,.21],[.22,.22],[.51,.51],[.52,.52]]
#     ]

    weights = [
        [np.random.rand(10)],
        [np.random.rand(10) for _ in range(10)],
        # [np.random.rand(10) for _ in range(100)]
    ]

    # weights = [[[]]]

    t1 = time.perf_counter()
    neuron = Neuron(
        name = 'neuron_0',
        threshold = 0.5,
        weights   = weights)
    t2 = time.perf_counter()

    neuron_2 = Neuron(
        name = 'neuron_1',
        threshold = 0.5,
        weights   = weights)

    for syn in neuron_2.synapse_list:
        neuron.dend_soma.outgoing.append(syn)
        syn.incoming.append(neuron.dend_soma)

    # print(f"Neuron creation time = {t2-t1}")
    print(f"Initial neuron object size = {sys.getsizeof(neuron)}\n")

    inpt_spike_times  = np.arange(0,50,100)

    t1 = time.perf_counter()
    neuron.add_uniform_input(inpt_spike_times)
    t2 = time.perf_counter()

    # print(f"Uniform input connection time = {t2-t1}")
    # print(type(neuron.synapse_list[0]))
    # neuron.dend_soma.offset_flux = 0.5


    t1 = time.perf_counter()
    net = Network(nodes=[neuron,neuron_2],run_simulation=True,multithreading=True,duration=50)
    t2 = time.perf_counter()
    print(f"Run time = {t2-t1}")

    # for node in net.nodes:
    #     for syn in node.synapse_list:
    #         plt.plot(syn.flux)
    # plt.show()



    plt.style.use("seaborn-v0_8-darkgrid")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axs = plt.subplots(len(net.nodes), 1,figsize=(12,4*len(net.nodes)))

    for n,node in enumerate(net.nodes):
        print(n,node.name)
        for dend in node.dendrite_list[2:]:
            axs[n].plot(dend.flux,'--')
            axs[n].plot(dend.signal)
        
        axs[n].plot(node.dend_ref.flux,':',color=colors[3])
        axs[n].plot(node.dend_soma.signal,linewidth=4,color=colors[0])
        axs[n].plot(node.dend_soma.flux,linewidth=2,color=colors[1])
        axs[n].set_title(node.name,fontsize=14)

    fig.text(0.5, .95, 
            f"Networking with slim_soens :: {net.nodes[0].name} to {net.nodes[1].name}", 
            ha='center',fontsize=18)
    fig.text(0.5, 0.05, 'Time (ns)', ha='center', fontsize=16)
    fig.text(0.05, 0.5, 'Signal (unitless)', va='center', rotation='vertical', fontsize=16)
    plt.show()


    # outgoing_map(neuron)
    # incoming_map(neuron)
    # print_names(neuron.synapse_list)
    # print_attrs(neuron.synapse_list,["name","spike_times"])


    print(f"\nFinal neuron object size = {sys.getsizeof(neuron)}\n")

if __name__=='__main__':
    main()