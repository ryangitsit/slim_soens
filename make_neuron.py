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

        self.dend_soma = Soma()
        self.dend_ref  = Refractory()
        self.dend_ref.outgoing = self.dend_soma

        self.dendrite_list += [self.dend_soma,self.dend_ref]
        self.make_components_comp()

    def find_dend(self,dend,coords):
        """
        Docstring
        """
        l,g,d = coords
        if f"{l}_{g}_{d}" in dend.name:


    def make_weighted_dend(self,l,g,d,weights,dend_params):
        """
        Docstring
        """
        if l==0:
            dend_params["outgoing"] = (self.dend_soma,weights[l][g][d])
        else:
            dend_params["outgoing"] = ((l-1,0,g),weights[l][g][d])
        dend_params["name"] = f"{self.name}_dend_{l}_{g}_{d}"
        dendrite = Dendrite(**dend_params)
        return dendrite
    

    def make_components_comp(self):
        """
        Docstring
        """
        dend_params = {}
        self.dendrite_list += [
            self.make_weighted_dend(l,g,d,self.weights,dend_params)
            for l,layer in enumerate(self.weights)
            for g,group in enumerate(layer)
            for d,dend in enumerate(group)
            ]
        

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
    
