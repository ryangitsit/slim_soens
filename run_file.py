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

def create_neuron():
    return Neuron(weights=weights_arbor)

t1 = time.perf_counter()
neuron = create_neuron()
t2 = time.perf_counter()

# print(f"Neuron creation time = {t2-t1}")
print(f"Initial neuron object size = {sys.getsizeof(neuron)}\n")

inpt_spike_times  = np.arange(10,500,50)

t1 = time.perf_counter()
neuron.add_uniform_input(inpt_spike_times)
t2 = time.perf_counter()

# print(f"Uniform input connection time = {t2-t1}")


net = Network(nodes=[neuron],run_simulation=True,duration=500)

for node in net.nodes:
    for syn in node.synapse_list:
        plt.plot(syn.flux)
plt.show()
# outgoing_map(neuron)
# incoming_map(neuron)
# print_names(neuron.synapse_list)
# print_attrs(neuron.synapse_list,["name","spike_times"])


print(f"\nFinal neuron object size = {sys.getsizeof(neuron)}\n")

