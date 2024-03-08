import numpy as np
import matplotlib.pyplot as plt
import timeit
import time
import sys

from neuron import Neuron
from network import Network

from system_functions import *
from plotting import heatmap_adjacency


weights_arbor = [
    [[0.2,0.5]],
    [[0.2,0.2],[0.5,0.5]],
    [[.21,.21],[.22,.22],[.51,.51],[.52,.52]]
]

weights = [
    [np.random.rand(10)],
    [np.random.rand(10) for _ in range(10)],
    [np.random.rand(10) for _ in range(100)]
]

t1 = time.perf_counter()
neuron = Neuron(weights=weights)
t2 = time.perf_counter()

# print(f"Neuron creation time = {t2-t1}")
print(f"Initial neuron object size = {sys.getsizeof(neuron)}\n")

inpt_spike_times  = np.arange(10,1000,50)

t1 = time.perf_counter()
neuron.add_uniform_input(inpt_spike_times)
t2 = time.perf_counter()

# print(f"Uniform input connection time = {t2-t1}")
# print(type(neuron.synapse_list[0]))
# neuron.dend_soma.offset_flux = 0.5
t1 = time.perf_counter()
net = Network(nodes=[neuron],run_simulation=True,duration=1000)
t2 = time.perf_counter()
print(f"Run time = {t2-t1}")

# for node in net.nodes:
#     for syn in node.synapse_list:
#         plt.plot(syn.flux)
# plt.show()

for node in net.nodes:
    for dend in node.dendrite_list[2:]:
        plt.plot(dend.flux,'--')
        plt.plot(dend.signal)
    plt.plot(neuron.dend_ref.flux,':',color='red')
    plt.plot(neuron.dend_soma.signal,linewidth=2,color='b')
plt.show()

# outgoing_map(neuron)
# incoming_map(neuron)
# print_names(neuron.synapse_list)
# print_attrs(neuron.synapse_list,["name","spike_times"])


print(f"\nFinal neuron object size = {sys.getsizeof(neuron)}\n")

