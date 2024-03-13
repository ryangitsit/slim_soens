import numpy as np
import matplotlib.pyplot as plt
import timeit
import time
import sys

from neuron import Neuron
from network import Network
import components

from system_functions import *

def main():
    np.random.seed(10)
    weights = [
        [np.random.rand(3)],
        [np.random.rand(3) for _ in range(3)],
        # [np.random.rand(10) for _ in range(100)]
    ]

    neuron = Neuron(
        name = 'neuron_0',
        threshold = 0.5,
        weights   = weights)
    
    max_phi_received = 0.5
    max_s = 0.72
    for dend in neuron.dendrite_list:

        input_sum    = 0
        input_maxes = []
        for indend,w in dend.incoming:
            if isinstance(indend,components.Dendrite): 
                print(f"{dend.name} <- {indend.name} * {np.round(w,2)}")
                maxed =  max_s*w
                # print(f"  {maxed}")
                input_sum+=maxed
                input_maxes.append(maxed)

        print(input_sum)
        if input_sum > max_phi_received:
            for i,(indend,w) in enumerate(dend.incoming):
                dend.incoming[i][1] = w*input_maxes[i]/input_sum
                print(dend.incoming[i][1])


    
    for syn in neuron.synapse_list:
        # print(syn.outgoing)
        syn.outgoing[0].flux_offset = 0.5

    net = Network(
        nodes=[neuron],
        run_simulation=True,
        duration=500
        )
    
    # for dend in neuron.dendrite_list:
    #     plt.plot(dend.signal)
    # plt.show()

    # for dend in neuron.dendrite_list:
    #     plt.plot(dend.flux)
    # plt.show()

if __name__=='__main__':
    main()
