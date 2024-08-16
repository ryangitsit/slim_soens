from system_functions import *
from components import *
import components

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

        self.weights      = [[[]]]
        self.arbor_params = None

        self.synaptic_strength = 1
        self.no_synapses       = False

        self.__dict__.update(params)

        if self.arbor_params is None:
            self.arbor_params = [
                [[{} for w,weight in enumerate(group)] 
                for g,group in enumerate(layer)]
                for l,layer in enumerate(self.weights)]

        self.dendrite_list = []

        self.dend_soma = Soma(neuron_name=self.name,neuron=self,**params)
        self.dend_ref  = Refractory(**{"neuron_name":self.name},neuron=self)

        # moved to adjacency matrix
        # self.dend_ref.outgoing  = [[self.dend_soma,-0.85]]
        # self.dend_soma.incoming = [[self.dend_ref,-0.85]]

        self.dendrite_list += [self.dend_soma,self.dend_ref]
        
        if hasattr(self,"arbor_adjacency"):
            self.make_dendrites_from_adj()
        else: 
            self.make_dendrites()
            self.arbor_adjacency = weights_to_adjacency(self.weights)

        self.add_edges()

        if self.no_synapses != True:
            # print("Adding synaptic layer...")
            self.add_synaptic_layer()
        else:
            self.synapse_list = []

        self.get_dimensions()

        

    def make_dend(self,l,g,d,dend_params):
        """
        Docstring
        """
        dend_params["name"] = f"{self.name}_dend_{l}_{g}_{d}"
        dend_params["loc"]  = (l,g,d)
        dendrite = Dendrite(neuron=self,**dend_params)
        return dendrite
    
    def get_dend_name(self,dend):
        return dend.name
    
    def make_dendrites(self):
        """
        Docstring
        """
        self.dendrite_list += [
            self.make_dend(l+1,g,d,self.arbor_params[l][g][d])
            for l,layer in enumerate(self.weights)
            for g,group in enumerate(layer)
            for d,dend in enumerate(group)
            ]
        
    def make_dendrites_from_adj(self):
        self.dendrite_list += [
            Dendrite(
                name=f"dendrite_{i}",
                neuron=self
                ) for i in range(len(self.arbor_adjacency)-2)]

        
    def make_dend_dict(self):
        """
        Docstring
        """
        dend_names = [*map(self.get_dend_name,self.dendrite_list)]
        self.dend_dict = dict(zip(dend_names,self.dendrite_list))

    def add_edges(self):
        total_dendrites = len(self.dendrite_list)
        for i in range(total_dendrites):
            for j in range(total_dendrites):
                if self.arbor_adjacency[i][j] != 0:
                    
                    self.dendrite_list[i].outgoing.append(
                        [self.dendrite_list[j],self.arbor_adjacency[i][j]]
                        )
                    self.dendrite_list[j].incoming.append(
                        [self.dendrite_list[i],self.arbor_adjacency[i][j]]
                        )   

    def add_synapse(self,dend,cs):
        syn = Synapse(neuron=self,**{'dend_name':dend.name}) 
        dend.incoming.append((syn,cs))
        syn.outgoing.append(dend)
        return syn

    def add_synaptic_layer(self):
        if len(self.dendrite_list) <= 2:
            self.synapse_list = [self.add_synapse(self.dend_soma,self.synaptic_strength)]
        else:
            self.synapse_list = [
                self.add_synapse(dend,self.synaptic_strength)
                for d,dend in enumerate(self.dendrite_list) 
                if dend.incoming==[] and 'ref' not in dend.name
                ]
            
    def add_symmetric_synaptic_layer(self):
        sw = [-1,1]
        if len(self.dendrite_list) <= 2:
            self.synapse_list = [
                self.add_synapse(self.dend_soma,self.synaptic_strength),
                self.add_synapse(self.dend_soma,-1*self.synaptic_strength)]
        else:
            self.synapse_list = [
                self.add_synapse(dend,self.synaptic_strength*sw[d%2])
                for d,dend in enumerate(self.dendrite_list) 
                if dend.incoming==[] and 'ref' not in dend.name
                ]
            
        
    def add_spikes(self,syn,spike_times):
        syn.spike_times = np.sort(np.concatenate([syn.spike_times,spike_times]))

    def add_uniform_input(self,spike_times):

        if type(spike_times)!=np.ndarray: spike_times = np.array(spike_times)
        
        for syn in self.synapse_list:
            syn.spike_times = np.sort(np.concatenate([syn.spike_times,spike_times]))

    

    def add_spike_rows(self,spike_rows):
        for i,row in enumerate(spike_rows):
            self.synapse_list[i].spike_times = np.sort(
                np.concatenate([self.synapse_list[i].spike_times,row])
                )
            
    def add_spike_rows_doubled(self,spike_rows):
        for i,row in enumerate(spike_rows):
            self.synapse_list[i*2].spike_times = np.sort(
                np.concatenate([self.synapse_list[i*2].spike_times,row])
                )
            self.synapse_list[i*2+1].spike_times = np.sort(
                np.concatenate([self.synapse_list[i*2+1].spike_times,row])
                )
            
    def add_indexed_spikes(self,indexed_spikes,channels=None,doubled=False):
        if not channels:
            channels = max(indexed_spikes[0])+1
        spike_rows = array_to_rows(indexed_spikes,channels)
        if doubled==False:
            self.add_spike_rows(spike_rows)
        else:
            self.add_spike_rows_doubled(spike_rows)



    def change_weight(self,dend,i,norm_factor):
        dend.incoming[i][1] *= norm_factor

        for o,out in enumerate(dend.incoming[i][0].outgoing):
            # if dend.incoming[i][0].outgoing[o].
            if out[0].name == dend.name:
                out[1] = dend.incoming[i][1]


    def normalize_fanin(self,fanin_factor=1):
        max_phi_received = 0.5
        max_s = 0.72
        for dend in self.dendrite_list:

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
                norm_ratio = fanin_factor*max_phi_received/input_sum
                [
                    self.change_weight(dend,i,norm_ratio) 
                    for i in range(len(dend.incoming)) 
                    if not isinstance(dend.incoming[0],components.Refractory)
                    ]
                
    def normalize_fanin_symmetric(self,fanin_factor=1):
        max_phi_received = 0.5
        max_s = 0.72
        for dend in self.dendrite_list:

            input_sum   = 0
            input_sum_negative = 0
            for indend,w in dend.incoming:
                if (isinstance(indend,components.Dendrite) 
                    and not isinstance(indend,components.Refractory)): 
                    if w > 0:
                        input_sum+=max_s*w
                    else:
                        input_sum_negative+=max_s*w
            # print(dend.name,input_sum,input_sum_negative)
            if input_sum!=0: #input_sum > max_phi_received:
                norm_ratio = fanin_factor*max_phi_received/input_sum
                [
                    self.change_weight(dend,i,norm_ratio) 
                    for i in range(len(dend.incoming)) 
                    if not isinstance(dend.incoming[i][0],components.Refractory)
                    and dend.incoming[i][1] > 0
                    ]
            if input_sum_negative!=0: #f np.abs(input_sum_negative) > max_phi_received:
                # norm_ratio_neg = fanin_factor*max_phi_received/np.abs(input_sum_negative)
                norm_ratio_neg = fanin_factor*0.1675/np.abs(input_sum_negative)
                [
                    self.change_weight(dend,i,norm_ratio_neg) 
                    for i in range(len(dend.incoming)) 
                    if not isinstance(dend.incoming[i][0],components.Refractory)
                    and dend.incoming[i][1] < 0
                    ]
                
    def make_photonic_connection_to(self,connection_obj,strength=1):
        """
        Makes an outgoing connection from soma to a synapse.
         - If connection object is a neuron, add a synapse to soma and connect to that.
         - If connection object is a dendrite, add a synapse and connect to that.
        """

        if type(connection_obj)==type(self):

            new_synapse = Synapse(
                name      = f"synapse_from_{self.name}",
                dend_name = connection_obj.dend_soma.name
                )
            connection_obj.dend_soma.incoming.append([new_synapse,strength])
            connection_obj.synapse_list.append(new_synapse)

            synapse = new_synapse

        elif (isinstance(connection_obj,components.Soma) 
              or isinstance(connection_obj,components.Dendrite)):
            new_synapse = Synapse(
                name=f"{connection_obj.name}_syn_from{self.name}",
                dend_name = connection_obj.name
                )
            connection_obj.incoming.append([new_synapse,strength])

            connection_obj.neuron.synapse_list.append(new_synapse)
            
            synapse = new_synapse

        elif isinstance(connection_obj,components.Synapse):
            synapse = connection_obj

        else:
            raise Exception("Invalid photonic connection type")
        
        make_directed_connection(self.dend_soma,synapse,cs=1)


    def randomize_offsets(self,radius):
        for dend in self.dendrite_list[2:]:
            dend.flux_offset = np.random.rand()*radius*np.random.choice([-1,1], p=[.5,.5], size=1)[0]

    def get_dimensions(self):
        dims = [2]
        for w  in self.weights:
            dims.append(count_total_elements(w))
        self.dims = dims
        self.layers = len(dims)
        return dims
    
    def parameter_print(self):
        print(f"Neuron {self.name} parameters:")
        for i,(k,v) in enumerate(self.__dict__.items()):
            print("  ",i,k," "*(20-len(k)),v)

    def plot_structure(self):
        from plotting import graph_adjacency
        graph_adjacency(self.arbor_adjacency,self.get_dimensions())



    def __del__(self):
        """
        """
        # print(f'Neuron {self.name} deleted')
        return
    