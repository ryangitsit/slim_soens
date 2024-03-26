import numpy as np
import matplotlib.pyplot as plt
import timeit
import time
import sys
sys.path.append('../')

from neuron import Neuron
from network import Network
import components

from system_functions import *

from simulate_multithrd import *

import multiprocessing as mp

NWORKERS     = 3
NTHREADS     = NWORKERS+1  # main thread + worker threads
NGENERATIONS = 10

def work_fun(barrier, tag):
    for generation in range(NGENERATIONS):
        barrier.wait()
        print(f"I am worker{tag}, generation={generation}", flush=True)

def update_node(node,t,dt,d_tau,tf):
    for dend in node.dendrite_list[1:]:
        update_dendrite(dend,t,dt,d_tau)
    update_soma(node.dend_soma,node.dend_ref,t,dt,d_tau,tf)
    return node

def run_parallel(barrier, tag, node, return_dict):
    tf = NGENERATIONS
    dt = 1
    d_tau = dt*1e-9/1.2827820602389245e-12
    for t in range(NGENERATIONS):
        if t < NGENERATIONS-1:
            node = update_node(node,t,dt,d_tau,tf)
        else:
            for dend in node.dendrite_list:
                update_flux(dend,t)
        return_dict[node.name] = node
        barrier.wait()
        print(f"I am worker{tag}, generation={t} => {node.dend_soma.flux[t]}", flush=True)

if __name__=='__main__':
    ctx = mp.get_context('spawn')
    barrier = ctx.Barrier(NTHREADS)
    procs = []

    manager = mp.Manager()
    return_dict = manager.dict()

    weights = [
        [np.random.rand(10)],
        [np.random.rand(10) for _ in range(10)],
        [np.random.rand(10) for _ in range(100)],
    ]

    node_count = 3

    nodes = []
    for n in range(node_count):
        nodes.append(
            Neuron(
                name    = f'node_{n}',
                weights = weights
            )
        )

    for node in nodes:
        for syn in node.synapse_list:
            syn.outgoing[0].flux_offset = 0.5

    tf = time_steps = NGENERATIONS
    dt = 1
    for node in nodes:
        initialize_synapses (node,tf,dt,time_steps)
        initialize_dendrites(node,tf,dt,time_steps)

    for i in range(NWORKERS):
        proc = ctx.Process(
            target=run_parallel, args=(barrier, i, nodes[i], return_dict))
        proc.start()
        procs.append(proc)


    for i in range(NGENERATIONS):
        time.sleep(1)
        barrier.wait()

    for i in range(NWORKERS):
        procs[i].join()

    return_nodes = []
    for n in range(len(nodes)):
        return_nodes.append(return_dict[f'node_{n}'])

    for node in return_nodes:
        print(f"Node {node.name} => {node.dend_soma.signal}")

