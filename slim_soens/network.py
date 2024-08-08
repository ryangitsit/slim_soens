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
        self.__dict__.update(params)
        self.jjparams = get_jj_params()
        if self.run_simulation is True:
            self.run_network_simulation(duration=self.duration)

    def run_network_simulation(self,duration=100):
        self.duration=duration
        run_slim_soens(self)
        
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

        self.__dict__.update(params)
        self.network_adjacency = np.zeros((self.N,self.N))
        self.make_net()
        

    def make_net(self):
        self.nodes = []
        for i in range(self.N):
            self.nodes.append(Neuron(
                name = f"HopNode_{i}"
            ))

        for i in range(self.N):
            for j in range(self.N):
                if i!=j: 
                    self.network_adjacency[i][j] = self.connection_strengths
                    self.nodes[i].make_photonic_connection_to(self.nodes[j],strength=self.connection_strengths)
