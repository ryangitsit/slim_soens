import numpy as np

from simulate import run_slim_soens
# from simulate_multithrd import run_slim_soens_multi
from system_functions import get_jj_params
from neuron import Neuron
from components import *

class Network():
    """
    Docstring
    """
    next_id = 0
    def __init__(self, **params):
        self.dt = 1
        self.duration = 100
        self.epochs = 0
        self.run_simulation = None
        self.multithreading = None
        self.backend = "default"
        self.no_spikes = False
        self.ib = 1.8
        self.phi_th = 0.1675
        self.activation = "superconducting"
        self.__dict__.update(params)
        self.jjparams = get_jj_params()

        if self.run_simulation is True:
            self.run_network_simulation(duration=self.duration)


    def run_network_simulation(self,duration=100):
        self.duration=duration
        if self.backend=="default":
            run_slim_soens(self)
        elif self.backend == "steady_state":
            from steady_state_backend import run_steady_state
            run_steady_state(self)

        
    def get_output_spikes(self):
        spike_times   = []
        spike_indices = []
        for i,node in enumerate(self.nodes):
            for spk in node.dend_soma.spikes:
                spike_times.append(spk)
                spike_indices.append(i)

        self.output_spikes = [spike_indices,spike_times]
        return self.output_spikes

    def run_network_simulation_multithread(self,duration=100):
        if self.backend == "default":
            run_slim_soens_multi(self)

    def plot_structure(self):
        pass

    def return_data(self):
        pass

    def clear_net(self):
        for node in self.nodes:
            node.dend_soma.spikes = []
            node.dend_soma.quiescence = 0
            for dend in node.dendrite_list:
                dend.signal = np.array([])
                dend.flux   = np.array([])

            for syn in node.synapse_list:
                syn.flux        = np.array([])
                syn.spike_times = np.array([])

            self.output_spikes = []
        del(self)


class HopfieldNetwork(Network):

    def __init__(self, **params):
        super().__init__()
        self.N                    = 10
        self.connection_strengths = 1
        self.global_offset = 0
        self.__dict__.update(params)
        self.make_net()
        

    def make_net(self):

        if hasattr(self,'network_adjacency') == False:
            self.network_adjacency = np.ones((self.N,self.N))*self.connection_strengths
        else:
            self.N = len(self.network_adjacency)

        self.nodes = []
        for i in range(self.N):
            neuron = Neuron(name = f"HopNode_{i}")
            neuron.dend_soma.flux_offset = self.global_offset
            self.nodes.append(neuron)

        for i in range(self.N):
            for j in range(self.N):
                self.nodes[i].make_photonic_connection_to(
                    self.nodes[j],
                    strength=self.network_adjacency[i][j]
                    )
