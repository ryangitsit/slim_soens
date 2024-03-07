import numpy as np
from make_neuron import *
from system_functions import *
import matplotlib.pyplot as plt
from plotting import heatmap_adjacency
import timeit
import time
import sys

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

print(f"Neuron creation time = {t2-t1}")
print(f"Initial neuron object size = {sys.getsizeof(neuron)}")

inpt_spike_times  = np.arange(10,500,50)

t1 = time.perf_counter()
neuron.add_uniform_input(inpt_spike_times)
t2 = time.perf_counter()

print(f"Uniform input connection time = {t2-t1}")


# outgoing_map(neuron)
# incoming_map(neuron)
# print_names(neuron.synapse_list)
# print_attrs(neuron.synapse_list,["name","spike_times"])


print(f"Final neuron object size = {sys.getsizeof(neuron)}")

