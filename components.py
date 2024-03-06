import numpy as np

class Synapse():
    """
    Docstring
    """
    next_id = 0
    def __init__(self, **params):

        # naming ***test this across multiple neurons
        self.id         = Synapse.next_id
        self.name       = f"Synapse_{self.id}"
        Synapse.next_id += 1

        # parameters
        self.tau_rise         = 0.02
        self.tau_fall         = 50
        self.hotspot_duration = 3
        self.spd_duration     = 8
        self.phi_peak         = 0.5
        self.spd_reset_time   = 35

        # connections
        self.incoming = None
        self.outgoing = None

    def add_outout(self,output_object):
        """
        Docstring
        """
        pass

    def __del__(self):
        print(f'Synapse {self.name} deleted')
        return


class Dendrite():
    """
    Docstring
    """
    next_id = 0
    def __init__(self, **params):

        # naming
        self.id         = Dendrite.next_id
        self.name       = f"Dendrite_{self.id}"
        Dendrite.next_id += 1

        # parameters
        self.ib   = 1.8
        self.tau  = 150
        self.beta = 2*np.pi*1e2

        # dynamic parameters
        self.offset_flux = 0

        # connections
        self.outgoing = None
        self.incoming = None

        self.__dict__.update(params)

    def add_input(self, input_object):
        """
        Docstring
        """
        pass

    def __del__(self):
        """
        """
        # print(f'Dendrite {self.name} deleted')
        return


class Soma(Dendrite):
    """
    Docstring
    """
    def __init__(self, **params):
        super().__init__()
        self.name       = f"Dendrite_{self.id}_Soma"
        self.__dict__.update(params)


    def __del__(self):
        """
        """
        # print(f'Somatic dendrite {self.name} deleted')
        return

class Refractory(Dendrite):
    """
    Docstring
    """
    def __init__(self, **params):
        super().__init__()
        self.name       = f"Dendrite_{self.id}_Refractory"
        self.__dict__.update(params)

    def __del__(self):
        """
        """
        # print(f'Refractory dendrite {self.name} deleted')
        return




