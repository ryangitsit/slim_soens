import numpy as np
from make_neuron import *
from system_functions import *
import matplotlib.pyplot as plt
from plotting import heatmap_adjacency
import timeit
import sys

weights_arbor = [
    [[0.2,0.5]],
    [[0.2,0.2],[0.5,0.5]],
    [[.21,.21],[.22,.22],[.51,.51],[.52,.52]]
]

def create_neuron():
    return Neuron(weights=weights_arbor)


neuron = create_neuron()

print(f"Neuron object size = {sys.getsizeof(neuron)}")



outgoing_map(neuron)

incoming_map(neuron)

# 
# # """
# # Try:
# #     weights -> adjacency -> make_dends -> connect_dends
# # """