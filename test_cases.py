import numpy as np
import matplotlib.pyplot as plt
import timeit
import time
import sys

from neuron import Neuron
from network import Network
import components
from plotting import *

from system_functions import *

def main():

    def test_normalization():
        np.random.seed(10)
        t1 = time.perf_counter()
        weights = [
            [np.random.rand(3)*-1],
            [np.random.rand(3)*-1 for _ in range(3)],
        ]
        # weights = [
        #     [np.ones(3)],
        #     [np.ones(3) for _ in range(3)],
        # ]
        # weights = [
        #     [np.ones(3)*-1],
        #     [np.ones(3)*-1 for _ in range(3)],
        # ]
        # weights = [
        #     [np.random.rand(3)],
        #     [np.random.rand(3) for _ in range(3)],
        # ]
        # weights = [
        #     [np.random.rand(10)],
        #     [np.random.rand(10) for _ in range(10)],
        #     [np.random.rand(10) for _ in range(100)],
        #     # [np.random.rand(10) for _ in range(1000)]
        # ]
        t2 = time.perf_counter()
        print(f"Time to conjur weights = {np.round(t2-t1,2)}")

        t1 = time.perf_counter()
        neuron = Neuron(
            name = 'neuron_0',
            threshold = 0.5,
            weights   = weights)
        t2 = time.perf_counter()
        print(f"Time make neuron = {np.round(t2-t1,2)}")
        neuron.normalize_fanin_symmetric()

        print_attrs(neuron.dendrite_list,['name','incoming'])

        for syn in neuron.synapse_list:
            # print(syn.outgoing)
            syn.outgoing[0].flux_offset = 0.5

        net = Network(
            nodes=[neuron],
            run_simulation=True,
            duration=500
            )
        
        plot_nodes([neuron])
        
        for dend in neuron.dendrite_list:
            plt.plot(dend.signal,label=dend.name)
        plt.legend()
        plt.show()

        for dend in neuron.dendrite_list:
            plt.plot(dend.flux,label=dend.name)
        plt.legend()
        plt.show()
    test_normalization()

if __name__=='__main__':
    main()
