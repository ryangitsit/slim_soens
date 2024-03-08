from simulate import run_slim_soens

class Network():
    """
    Docstring
    """
    next_id = 0
    def __init__(self, **params):
        self.dt = 1
        self.duration = 100
        self.run_simulation = None
        self.__dict__.update(params)
        if self.run_simulation == True:
            self.run_network_simulation()

    def run_network_simulation(self):
        run_slim_soens(self)

    def plot_structure(self):
        pass

    def return_data(self):
        pass