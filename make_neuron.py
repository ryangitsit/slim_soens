from components import *

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
        self.dend_ref  = Refractory()
        self.dend_ref.outgoing = self.dend_soma

        self.dendrite_list += [self.dend_soma,self.dend_ref]
        
        self.make_dendrites()
        self.make_dend_dict()
        self.connect_dendrites()

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

    
    def add_arbor_vertex(self,dend):
        """
        Docstring
        """

        out_loc = (dend.loc[0]-1,0,dend.loc[1])
        print(f"{dend.loc} -> {out_loc}")
        if dend.loc[0]==1:
            out_name = f"{self.name}_dend_soma"
        else:
            out_name = f"{self.name}_dend_{out_loc[0]}_{out_loc[1]}_{out_loc[2]}"
        dend.outgoing = self.dend_dict[out_name]
        return f"{dend.loc} -> {out_loc}"

    def connect_dendrites(self):
        connect = map(self.add_arbor_vertex,self.dendrite_list[2:])
        print([*connect])

    def make_components_loop(self):
        """
        Docstring
        """
        for l,layer in enumerate(self.weights):
            for g,group in enumerate(layer):
                for d,dend in enumerate(group):
                    dendrite = Dendrite()
                    self.dendrite_list.append(dendrite)

    def connect_denrites(self):
        """
        Docstring
        """
        count = 0
        for l,layer in enumerate(self.weights):
            for g,group in enumerate(layer):
                for d,dend in enumerate(group):
                    
                    if l==0:
                        self.dendrite_list[count+2].outgoing = self.dend_soma
                    
                    # else:
                    #     self.dendrite_list[count+2].outgoing = 
                    
                    count+=1


    def __del__(self):
        """
        """
        # print(f'Neuron {self.name} deleted')
        return
    