#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append('../')

from system_functions import *
from plotting import *
from neuron import Neuron
from network import Network
import components

#%%
exp_name = "mid_mnist_2"
digits = 10
samples = 1000

# exp_name = "cifar_small"
# digits = 3
# samples = 10
readout_nodes = picklin(f"../results/mnist_study/{exp_name}/",f"readouts_symmetric_{digits}_{samples}_0_at_0")

#%%
print(len(readout_nodes))
# %%
for i,node in enumerate(readout_nodes):
    print(sys.getsizeof(node))
    sizeup_obj(node)
#%%

print(node.dendrite_list[10].__dict__.keys())
print(sys.getsizeof(node.dendrite_list[10].update_traj))
# print(len(node.dendrite_list))
# sizeup_obj(node.dendrite_list[0])

traj = node.dendrite_list[10].update_traj
print(type(traj))
traj_arr = np.array(traj)
print(type(traj_arr))
print(sys.getsizeof(traj_arr))
#%%



def plot_representations(nodes,shape=(28,28),rgb_dims=1):

    fig,ax = plt.subplots(len(nodes),2*rgb_dims,figsize=(6,len(nodes)), sharex=True,sharey=True)
    
    for n,node in enumerate(nodes):
        
        learned_offsets_positive = []
        learned_offsets_negative = []

        for i,dend in enumerate(node.dendrite_list[2:]):
            if dend.outgoing[0][1] >= 0:
                learned_offsets_positive.append(dend.flux_offset)
            else:
                learned_offsets_negative.append(dend.flux_offset)
        sub_arrlen = int(len(learned_offsets_positive)/rgb_dims)
        for i in range(rgb_dims):
            start = i*sub_arrlen
            stop = i*sub_arrlen+sub_arrlen
            j1 = i*2
            j2 = i*2+1
            ax[n][j1].imshow(np.array(learned_offsets_positive)[start:stop].reshape(shape),cmap="Greens")
            ax[n][j2].imshow(np.array(learned_offsets_negative)[start:stop].reshape(shape),cmap="Reds")
            ax[n][j1].set_xticks([])
            ax[n][j2].set_yticks([])

    plt.show()
plot_representations(readout_nodes,shape=(32,32),rgb_dims=3)
# %%

# dct = {
#     "name":"rain",
#     "date":1000
# }

# def print_fun(name="johndoe",date=0):
#     print(name)
#     print(date)

# print_fun(**dct)