from components import *

class Neuron():
    """
    """
    next_id = 0
    def __init__(self, **params):

        # naming 
        self.id         = Neuron.next_id
        self.name       = f"Neuron_{self.id}"
        Dendrite.next_id += 1

        self.__dict__.update(params)

        self.dendrite_list = []

        self.dend_soma = Soma()
        self.dend_ref  = Refractory()
        self.dend_ref.outgoing = self.dend_soma

        self.dendrite_list += [self.dend_soma,self.dend_ref]
        
        self.make_components_comp()
        # self.make_components_loop()

        self.connect_denrites()

    def make_components_loop(self):

        for l,layer in enumerate(self.weights):
            for g,group in enumerate(layer):
                for d,dend in enumerate(group):
                    dendrite = Dendrite()
                    self.dendrite_list.append(dendrite)

    def make_components_comp(self):

        self.dendrite_list += [
            Dendrite() 
            for l,layer in enumerate(self.weights) 
            for g,group in enumerate(layer) 
            for d,dend in enumerate(group) 
            ]

    def connect_denrites(self):
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
    
