import numpy as np
import matplotlib.pyplot as plt
import time
import sys

from neuron import Neuron
from network import Network

from system_functions import *

def backend_timer_duration(backends,durations):
    """
    Creates and runs a simple 'point' neuron with periodic spiketrain
    """

    np.random.seed(10)

    # def_spikes = np.arange(0,500,100)
    # inpt = SuperInput(channels=1, type='defined', defined_spikes = def_spikes)


    # durations = np.arange(1000,100001,1000)
    # durations = np.arange(100,501,100)
    # print(len(durations))

    
    run_times_per_duration = [[],[]]
    variance_per_duration = [[],[]]
    for b,backend in enumerate(backends):
        for d,duration in enumerate(durations):
            run_times = []
            for i in range(11):

                node = Neuron(threshold=10)
                node.dendrite_list[0].offset_flux = 0.5

                net = Network(
                    run_simulation=True,
                    dt=.1,
                    duration=duration,
                    nodes=[node],)
                
                run_time = net.run_time

                if i != 0: run_times.append(run_time)

                print(
                    f"{backend} backend - duration {duration} - iteration {i} - run time {run_time}   ",
                    end="\r"
                    )

                del(node)
                del(net)
            run_times_per_duration[b].append(np.mean(run_times))
            variance_per_duration[b].append(np.std(run_times))
    runtime_data = [run_times_per_duration,variance_per_duration]
    picklit(runtime_data,"results/profiling/",f"runtimes_pointneuron_{backend}")
    return runtime_data

# durations = np.arange(1000,50001,1000)

# backends = ['slim_soens']
# runtime_data = backend_timer_duration(backends,durations)

def plot_pointneuron_runtimes(runtime_data,durations):
    plt.style.use("seaborn-v0_8-darkgrid")

    plt.figure(figsize=(8,4))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for b,backend in enumerate(backends):
        plt.plot(durations,runtime_data[0][b],linewidth=2,label=backend,color=colors[b])
        
        lower_bound = runtime_data[0][b]-0.5*np.array(runtime_data[1][b])
        upper_bound = runtime_data[0][b]+0.5*np.array(runtime_data[1][b])

        plt.fill_between(durations, lower_bound, upper_bound, 
                        facecolor=colors[b], alpha=0.2)
        
    plt.title("Time Stepper Run Time for Single Dendrite",fontsize=16)
    plt.xlabel("Duration (ns)",fontsize=14)
    plt.ylabel("Run Time (s)",fontsize=14)
    plt.legend()
    plt.savefig(f"results/profiling/runtimes_point_neuron_{backend}")
    plt.show()



def backend_timer_size(backends,layers):
    """
    Creates and runs a simple 'point' neuron with periodic spiketrain
    """

    np.random.seed(10)


    run_times_per_layer = [[],[]]
    variance_per_layer= [[],[]]
    
    for b,backend in enumerate(backends):
        for l,layer in enumerate(layers):
            run_times = []
            for i in range(11):
                
                weights = binary_fanin(layer)

                node = Neuron(weights=weights,threshold=10)
                node.dendrite_list[0].offset_flux = 0.5

                net = Network(
                    run_simulation=True,
                    dt=.1,
                    duration=1000,
                    nodes=[node],)
                run_time = net.run_time

                if i != 0: 
                    run_times.append(run_time)


                print(
                    f"{backend} backend - layers {layer} - iteration {i} - run time {net.run_time}   ",
                    end="\r"
                    )

                del(node)
                del(net)
            run_times_per_layer[b].append(np.mean(run_times))
            variance_per_layer[b].append(np.std(run_times))

    runtime_data = [run_times_per_layer,variance_per_layer]
    picklit(runtime_data,"results/profiling/",f"runtimes_layers_{backend}")
    return runtime_data


layers = np.arange(2,10,1).astype('int32')
backends = ['slim_soens']
# runtime_data = backend_timer_size(backends,layers)

def plot_size_runtimes(runtime_data,layers):
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize=(8,4))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for b,backend in enumerate(backends):
        plt.plot(layers,runtime_data[0][b],linewidth=2,label=backend,color=colors[b])
        
        lower_bound = runtime_data[0][b]-0.5*np.array(runtime_data[1][b])
        upper_bound = runtime_data[0][b]+0.5*np.array(runtime_data[1][b])

        plt.fill_between(layers, lower_bound, upper_bound, 
                        facecolor=colors[b], alpha=0.2)


    plt.title("Time Stepper Run Time for Increasing Arbor size",fontsize=16)
    plt.xlabel(r"Layers of Binary Fanin $N=2^{layers}$",fontsize=14)
    plt.ylabel("Run Time (s)",fontsize=14)
    plt.legend()
    plt.savefig(f"results/profiling/runtimes_arbor_{backend}")
    plt.show()
runtime_data = picklin("results/profiling/","runtimes_layers_slim_soens")
plot_size_runtimes(runtime_data,layers)