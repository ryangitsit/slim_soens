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

        self.__dict__.update(params)

        self.dendrite_list = []

        self.dend_soma = Soma(**{"neuron_name":self.name})
        self.dend_ref  = Refractory(**{"neuron_name":self.name})
        self.dend_ref.outgoing = [self.dend_soma]

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
                    self.dendrite_list[i].outgoing.append(self.dendrite_list[j])
                    self.dendrite_list[j].incoming.append(self.dendrite_list[i])   

    def add_synapse(self,dend):
        syn = Synapse(**{'dend_name':dend.name}) 
        dend.incoming = syn
        return syn

    def add_synaptic_layer(self):
        self.synapse_list = [
            self.add_synapse(dend)
            for d,dend in enumerate(self.dendrite_list) 
            if dend.incoming==[] and 'ref' not in dend.name
            ]
        
    def add_spikes(self,syn,spike_times):
        syn.spike_times = np.sort(np.concatenate([syn.spike_times,spike_times]))

    def add_uniform_input(self,spike_times):

        if type(spike_times)!=np.ndarray: spike_times = np.array(spike_times)

        # spike_adder = [
        #     self.add_spikes(syn,spike_times)
        #     for syn in self.synapse_list
        #     ]
        
        for syn in self.synapse_list:
            syn.spike_times = np.sort(np.concatenate([syn.spike_times,spike_times]))



    # def add_arbor_vertex(self,dend):
    #     """
    #     Docstring
    #     """

    #     out_loc = (dend.loc[0]-1,0,dend.loc[1])
    #     print(f"{dend.loc} -> {out_loc}")
    #     if dend.loc[0]==1:
    #         out_name = f"{self.name}_dend_soma"
    #     else:
    #         out_name = f"{self.name}_dend_{out_loc[0]}_{out_loc[1]}_{out_loc[2]}"
    #     dend.outgoing = self.dend_dict[out_name]
    #     return f"{dend.loc} -> {out_loc}"

    # def connect_dendrites(self):
    #     connect = map(self.add_arbor_vertex,self.dendrite_list[2:])
    #     print([*connect])

    # def make_components_loop(self):
    #     """
    #     Docstring
    #     """
    #     for l,layer in enumerate(self.weights):
    #         for g,group in enumerate(layer):
    #             for d,dend in enumerate(group):
    #                 dendrite = Dendrite()
    #                 self.dendrite_list.append(dendrite)

    # def connect_denrites(self):
    #     """
    #     Docstring
    #     """
    #     count = 0
    #     for l,layer in enumerate(self.weights):
    #         for g,group in enumerate(layer):
    #             for d,dend in enumerate(group):
                    
    #                 if l==0:
    #                     self.dendrite_list[count+2].outgoing = self.dend_soma
                    
    #                 # else:
    #                 #     self.dendrite_list[count+2].outgoing = 
                    
    #                 count+=1


    def __del__(self):
        """
        """
        # print(f'Neuron {self.name} deleted')
        return
    