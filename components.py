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
        self.incoming = []
        self.outgoing = []

        self.spike_times = np.array([])
        self.flux        = np.array([])

        self.__dict__.update(params)
        self.name += "_to_"+self.dend_name

    def add_outout(self,output_object):
        """
        Docstring
        """
        pass

    def __del__(self):
        # print(f'Synapse {self.name} deleted')
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
        self.ib     = 1.8
        self.tau    = 150
        self.beta   = 2*np.pi*1e3
        self.alpha  = 0.053733049288045114
        self.phi_th = 1.675

        # dynamic parameters
        self.flux_offset = 0

        # for rollover counting
        self.high_roll = 0
        self.low_roll  = 0

        # connections
        self.outgoing = []
        self.incoming = []

        # collectors
        self.flux   = []
        self.signal = []

        self.__dict__.update(params)

        tau_di = self.tau * 1e-9  
        Ic = 100 * 1e-6
        Ldi = 6.62606957e-34/(2*1.60217657e-19)*self.beta/(2*np.pi*Ic)
        rdi = Ldi/tau_di
        self.alpha = rdi/2.565564120477849

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
        neuron_name     = params["neuron_name"]
        self.name       = f"{neuron_name}_dend_soma"
        self.loc        = (0,0,0)
        self.threshold  = 0.5
        self.quiescence = 0
        self.spikes     = []
        self.abs_ref    = 10
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
        neuron_name = params["neuron_name"]
        self.name   = f"{neuron_name}_dend_ref"
        self.loc        = (0,0,1)
        self.__dict__.update(params)

    def __del__(self):
        """
        """
        # print(f'Refractory dendrite {self.name} deleted')
        return




