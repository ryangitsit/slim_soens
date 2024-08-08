import numpy as np
import time
import components

def initialize_dendrites(node):
    """
    Docstring
    """
    leaf_dends = []
    for dend in node.dendrite_list:
        dend.updated = False
        dend.flux = dend.flux_offset
        dend.signal = steady_signal_update(dend.flux)
        if dend.flux > 0:
            leaf_dends.append(dend)
    return leaf_dends

def steady_signal_update(flux):
    return steady_signal_update.ssa[
        int(np.clip(flux,-.99,.99)*steady_signal_update.lens_ssa)
        ]

def recursive_flux_update(dend):
    for o,out in enumerate(dend.outgoing):
        out[0].flux = dend.signal*out[1]
        out[0].signal = steady_signal_update(out[0].flux)
        out[0].updated = True
        recursive_flux_update(out[0])

def run_steady_state(net):
    # print("STEADY STATE BACKEND")
    steady_signal_array = np.load("steady_signal.npy")
    steady_signal_update.ssa = steady_signal_array
    steady_signal_update.lens_ssa = len(steady_signal_array)
    for node in net.nodes:
        
        leaf_dends = initialize_dendrites(node)
        # print(len(leaf_dends))
        for l,leaf in enumerate(leaf_dends):
            recursive_flux_update(leaf)
    

