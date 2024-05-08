from simulate import run_slim_soens
from simulate_multithrd import run_slim_soens_multi
from system_functions import get_jj_params

class Network():
    """
    Docstring
    """
    next_id = 0
    def __init__(self, **params):
        self.dt = 1
        self.duration = 100
        self.run_simulation = None
        self.multithreading = None
        self.__dict__.update(params)
        self.jjparams = get_jj_params()
        if self.run_simulation is True:
            if self.multithreading is True:
                print("Multithreading")
                self.run_network_simulation_multithread()
            else:
                self.run_network_simulation()

    def run_network_simulation(self):
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

    def run_network_simulation_multithread(self):
        run_slim_soens_multi(self)

    def plot_structure(self):
        pass

    def return_data(self):
        pass