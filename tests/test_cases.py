import numpy as np
import matplotlib.pyplot as plt
import timeit
import time
import sys
sys.path.append('../')

from neuron import Neuron
from network import Network
import components
from plotting import *

from system_functions import *

def main():

    def test_rollover_regime():
        plt.style.use('seaborn-v0_8-muted')
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        
        # print(neuron.__dict__)
        # print_attrs(neuron.synapse_list,['names'])
        offs = np.arange(0,2,.2)
        signals = []
        fluxes  = []
        for i,off in enumerate(offs):
            neuron = Neuron(threshold=10)
            neuron.dend_soma.flux_offset = off
            net = Network(
                nodes=[neuron],
                run_simulation=True,
                duration=1000
                )
            signal = neuron.dend_soma.signal
            flux = neuron.dend_soma.flux
            signals.append(signal)
            fluxes.append(flux)
            plt.plot(signal,   color=colors[i%len(colors)],label=str(off))
            # plt.plot(flux,'--',color=colors[i%len(colors)])
            clear_net(net)
        plt.legend()
        plt.show()
    test_rollover_regime()


    def test_normalization():
        neurons = 10
        np.random.seed(10)
        t1 = time.perf_counter()
        # weights = [
        #     [np.random.rand(3)*-1],
        #     [np.random.rand(3)*-1 for _ in range(3)],
        # ]
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
        weights = [
            [np.random.rand(10)],
            [np.random.rand(10) for _ in range(10)],
            [np.random.rand(10) for _ in range(100)],
            # [np.random.rand(10) for _ in range(1000)]
        ]
        t2 = time.perf_counter()
        print(f"Time to conjur weights = {np.round(t2-t1,2)}")

        t1 = time.perf_counter()
        nodes = []
        for i in range(neurons):
            neuron = Neuron(
                name = f'neuron_{i}',
                threshold = 0.5,
                weights   = weights)
            neuron.normalize_fanin_symmetric()
            nodes.append(neuron)
        t2 = time.perf_counter()
        print(f"Time make neurons = {np.round(t2-t1,2)}")

        # print_attrs(neuron.dendrite_list,['name','incoming'])

        for node in nodes:
            for syn in node.synapse_list:
                # print(syn.outgoing)
                syn.outgoing[0].flux_offset = 0.5

        t1 = time.perf_counter()
        net = Network(
            nodes=nodes,
            run_simulation=True,
            duration=1000
            )
        t2 = time.perf_counter()
        print(f"Time run net = {np.round(t2-t1,2)}")
        
        plot_nodes(nodes)
        
        # for dend in neuron.dendrite_list:
        #     plt.plot(dend.signal,label=dend.name)
        # plt.legend()
        # plt.show()

        # for dend in neuron.dendrite_list:
        #     plt.plot(dend.flux,label=dend.name)
        # plt.legend()
        # plt.show()
    # test_normalization()

if __name__=='__main__':
    main()
