from components import *
from system_functions import *

class Neuron():
    """
    Docstring
    """
    next_id = 0
    def __init__(self, **params):

        # naming
        self.id         = Neuron.next_id
        self.name       = f"Neuron_{self.id}"
        Neuron.next_id += 1

        self.weights = [[[]]]
        self.synaptic_strength = 1

        self.__dict__.update(params)

        self.dendrite_list = []

        self.dend_soma = Soma(neuron_name=self.name,**params)
        self.dend_ref  = Refractory(**{"neuron_name":self.name})

        self.dend_ref.outgoing = [(self.dend_soma,-0.85)]
        self.dend_soma.incoming = [(self.dend_ref,-0.85)]

        self.dendrite_list += [self.dend_soma,self.dend_ref]
        
        self.make_dendrites()

        self.adjacency = weights_to_adjacency(self.weights)

        self.add_edges()

        self.add_synaptic_layer()

        

    def make_dend(self,l,g,d,dend_params):
        """
        Docstring
        """
        dend_params["name"] = f"{self.name}_dend_{l}_{g}_{d}"
        dend_params["loc"]  = (l,g,d)
        dendrite = Dendrite(**dend_params)
        return dendrite
    
    def get_dend_name(self,dend):
        return dend.name
    
    def make_dendrites(self):
        """
        Docstring
        """
        dend_params = {}
        self.dendrite_list += [
            self.make_dend(l+1,g,d,dend_params)
            for l,layer in enumerate(self.weights)
            for g,group in enumerate(layer)
            for d,dend in enumerate(group)
            ]
        
    def make_dend_dict(self):
        """
        Docstring
        """
        dend_names = [*map(self.get_dend_name,self.dendrite_list)]
        print(dend_names)
        self.dend_dict = dict(zip(dend_names,self.dendrite_list))
        

    def add_edges(self):
        total_dendrites = len(self.dendrite_list)
        for i in range(total_dendrites):
            for j in range(total_dendrites):
                if self.adjacency[i][j] != 0:
                    
                    self.dendrite_list[i].outgoing.append(
                        (self.dendrite_list[j],self.adjacency[i][j])
                        )
                    self.dendrite_list[j].incoming.append(
                        (self.dendrite_list[i],self.adjacency[i][j])
                        )   

    def add_synapse(self,dend):
        syn = Synapse(**{'dend_name':dend.name}) 
        dend.incoming.append((syn,self.synaptic_strength))
        return syn

    def add_synaptic_layer(self):
        if len(self.dendrite_list) <= 2:
            self.synapse_list = [self.add_synapse(self.dend_soma)]
        else:
            self.synapse_list = [
                self.add_synapse(dend)
                for d,dend in enumerate(self.dendrite_list) 
                if dend.incoming==[] and 'ref' not in dend.name
                ]
        
    def add_spikes(self,syn,spike_times):
        syn.spike_times = np.sort(np.concatenate([syn.spike_times,spike_times]))

    def add_uniform_input(self,spike_times):

        if type(spike_times)!=np.ndarray: spike_times = np.array(spike_times)
        
        for syn in self.synapse_list:
            syn.spike_times = np.sort(np.concatenate([syn.spike_times,spike_times]))



    def __del__(self):
        """
        """
        # print(f'Neuron {self.name} deleted')
        return
    