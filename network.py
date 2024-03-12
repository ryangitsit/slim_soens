from simulate import run_slim_soens
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
        self.multi_threading = None
        self.__dict__.update(params)
        self.jjparams = get_jj_params()
        if self.run_simulation == True:
            if self.multithreading==True:
                print(f"Multithreading")
                from simulate_multithrd import run_slim_soens_multi
                run_slim_soens_multi(self)
            else:
                self.run_network_simulation()
    
        

    def run_network_simulation(self):
        run_slim_soens(self)

    def plot_structure(self):
        pass

    def return_data(self):
        pass