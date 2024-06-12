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
exp_name = "mid_mnist"
readout_nodes = picklin(f"../results/mnist_study/{exp_name}/",f"readouts_symmetric_10_1000_0_at_0")

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
plot_representations(readout_nodes)
# %%
