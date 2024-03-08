import numpy as np

def generic_synaptic_event(dt):
    tau_rise         = 0.02/dt
    tau_fall         = 50/dt
    hotspot_duration = 3
    hotspot          = int(hotspot_duration/dt)
    spd_duration     = int(8*tau_fall)
    flux_peak        = 0.5
    
    flux_rise = [flux_peak*(1-np.exp(-t/tau_rise)) 
                 for t in range(hotspot)]

    flux_fall = [flux_peak*(1-np.exp(-hotspot/tau_rise))*np.exp(-(t-hotspot)/tau_fall) 
                 for t in range(hotspot,hotspot+spd_duration)]
    
    return np.concatenate([flux_rise,flux_fall])


def initialize_synapses(node,tf,dt,time_steps):
    flux_spd = generic_synaptic_event(dt)

    for syn in node.synapse_list:
        syn.flux = np.zeros((time_steps,))
        for spk_t in syn.spike_times:

            flux_final_idx = len(flux_spd)
            spike_time     = int(spk_t/dt)
            maxed          = flux_final_idx+spike_time
            spd_tf         = np.min([maxed,tf])
                
            syn.flux[spike_time:spd_tf] = np.maximum(
                flux_spd[:spd_tf-spike_time],
                syn.flux[spike_time:spd_tf])

def initialize_dendrites(node,tf,dt,time_steps):

    for dend in node.dendrite_list:
        dend.flux = np.zeros((time_steps,))




def run_slim_soens(net):
    time_steps = int(net.duration/net.dt)
    for node in net.nodes:
        initialize_synapses (node,net.duration,net.dt,time_steps)
        initialize_dendrites(node,net.duration,net.dt,time_steps)


    for node in net.nodes:
        print(node.name)