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
    t1 = time.perf_counter()
    weights = [
        [np.random.rand(3)],
        [np.random.rand(3) for _ in range(3)],
    ]
    weights = [
        [np.random.rand(10)],
        [np.random.rand(10) for _ in range(10)],
        [np.random.rand(10) for _ in range(100)],
        [np.random.rand(10) for _ in range(1000)]
    ]
    t2 = time.perf_counter()
    print(f"Time to conjur weights = {np.round(t2-t1,2)}")

    t1 = time.perf_counter()
    neuron = Neuron(
        name = 'neuron_0',
        threshold = 0.5,
        weights   = weights)
    t2 = time.perf_counter()
    print(f"Time make neuron = {np.round(t2-t1,2)}")

    def change_weight(dend,i,norm_factor):
        dend.incoming[i][1] *= norm_factor

    t1 = time.perf_counter()
    max_phi_received = 0.5
    max_s = 0.72
    for dend in neuron.dendrite_list:

        input_sum   = 0
        input_maxes = []
        for indend,w in dend.incoming:
            if (isinstance(indend,components.Dendrite) 
                and not isinstance(indend,components.Refractory)): 
                # print(f"{dend.name} <- {indend.name} * {np.round(w,2)}")
                maxed =  max_s*w
                # print(f"  {maxed}")
                input_sum+=maxed
                input_maxes.append(maxed)

        if input_sum > max_phi_received:
            norm_ratio = max_phi_received/input_sum
            [
                change_weight(dend,i,norm_ratio) 
                for i in range(len(dend.incoming)) 
                if not isinstance(dend.incoming[0],components.Refractory)
                ]

            # print(f"{dend.name} -> new sum = {new_sum}\n")
    
    t2 = time.perf_counter()
    print(f"Time to normalize = {np.round(t2-t1,5)}")

    for syn in neuron.synapse_list:
        # print(syn.outgoing)
        syn.outgoing[0].flux_offset = 0.5

    net = Network(
        nodes=[neuron],
        run_simulation=True,
        duration=500
        )
    
    for dend in neuron.dendrite_list:
        plt.plot(dend.signal)
    plt.show()

    for dend in neuron.dendrite_list:
        plt.plot(dend.flux)
    plt.show()

if __name__=='__main__':
    main()
