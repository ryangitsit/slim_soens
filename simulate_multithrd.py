import numpy as np
import time
import components
import multiprocessing

def generic_synaptic_event(dt):
    """
    Docstring
    """
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

def add_spike(syn,spk_t,dt,tf):
    """
    Docstring
    """
    
    flux_spd       = components.Synapse.flux_spd      
    flux_final_idx = components.Synapse.flux_final_idx

    spike_time     = int(spk_t/dt)
    maxed          = flux_final_idx+spike_time
    spd_tf         = np.min([maxed,tf])

    syn.flux[spike_time:spd_tf] = np.maximum(
        flux_spd[:spd_tf-spike_time],
        syn.flux[spike_time:spd_tf])
    
def initialize_synapses(node,tf,dt,time_steps):
    """
    Docstring
    """
    flux_spd = generic_synaptic_event(dt)
    flux_final_idx = len(flux_spd)
    
    components.Synapse.flux_spd       = flux_spd
    components.Synapse.flux_final_idx = len(flux_spd)

    for syn in node.synapse_list:
        syn.flux = np.zeros((time_steps,))
        for spk_t in syn.spike_times:

            add_spike(syn,spk_t,dt,tf)

def initialize_dendrites(node,tf,dt,time_steps):
    """
    Docstring
    """
    for dend in node.dendrite_list:
        dend.flux   = np.ones((time_steps,))*dend.offset_flux
        dend.signal = np.zeros((time_steps,))

def find_phi_th(val,A,B):
    """
    Docstring
    """
    return A*np.arccos(val/2) + B*(2-val)

# @jit(nopython=True)
def s_of_phi(phi,s,A=1,B=.466,ib=1.8):
    """
    Docstring
    """
    phi_th = 0.1675
    r_fq = A*(phi-phi_th)*(B*ib-s)
    if phi<phi_th: r_fq = 0
    return r_fq

def update_flux(dend,t):
    """
    Docstring
    """
    recieved_flux = 0
    for in_obj,w in dend.incoming:
        if type(in_obj) == components.Synapse:
            recieved_flux+=in_obj.flux[t]*w
        else:
            recieved_flux+=in_obj.signal[t]*w
    dend.flux[t] += recieved_flux

def update_signal(dend,t,dt,d_tau):
    """
    Docstring
    """
    r_fq = s_of_phi(dend.flux[t],dend.signal[t])
    dend.signal[t+1] = dend.signal[t] * ( 
            1 - d_tau*dend.alpha/dend.beta
            ) + (d_tau/dend.beta) * r_fq


def update_dendrite(dend,t,dt,d_tau):
    """
    Docstring
    """
    update_flux(dend,t)
    update_signal(dend,t,dt,d_tau)
    return dend

def update_soma(soma,ref,t,dt,d_tau,tf):
    """
    Docstring
    """
    update_flux(soma,t)
    if t >= soma.quiescence:
        if soma.signal[t] >= soma.threshold:
            soma.signal[t+1] = 0
            if t+10/dt < tf:
                for syn in soma.outgoing:
                    add_spike(syn,t+10/dt,dt,len(soma.signal))
                add_spike(ref,t+1,dt,len(soma.signal))
            soma.quiescence = t+10/dt
        else:
            update_signal(soma,t,dt,d_tau)
    return soma

def update_arbor(n,return_dict,node,t,dt,d_tau,tf):
    for dend in node.dendrite_list[1:]:
        update_dendrite(dend,t,dt,d_tau)

    update_soma(node.dend_soma,node.dend_ref,t,dt,d_tau,tf)
    return_dict[node.name] = node

def run_slim_soens_multi(net):
    """
    Docstring
    """
    time_steps = tf = int(net.duration/net.dt)
    d_tau = net.jjparams['t_tau_cnvt']*net.dt
    dt = net.dt

    for node in net.nodes:
        initialize_synapses (node,net.duration,net.dt,time_steps)
        initialize_dendrites(node,net.duration,net.dt,time_steps)

    

    t1 = time.perf_counter()

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    for t in range(time_steps-1):
        print(t)
        thrds = []
        for n,node in enumerate(net.nodes):

            thrds.append(
                multiprocessing.Process(
                    target=update_arbor,
                    args=(n,return_dict,node,t,dt,d_tau,tf)
                    )
                    )


        for thrd in thrds:
            thrd.start()

        for thrd in thrds:
            thrd.join()


        # for n in range(len(net.nodes)):
        #     net.nodes.append(return_dict[f'neuron_{n}'])

        del(thrds)

    t = time_steps-1
    for node in net.nodes:
        for dend in node.dendrite_list:
            update_flux(dend,t)
    t2 = time.perf_counter()
    net.run_time = t2-t1
