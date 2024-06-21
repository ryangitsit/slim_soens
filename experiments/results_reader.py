#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append('../')

from system_functions import *


#%%

def plot_training(path,train=True,test=False,val=False):
    plt.style.use('seaborn-v0_8-muted')
    plt.figure(figsize=(8,4))
    if train==True: 
        accs = np.array(picklin(path,'learning_accs'))*100
        plt.plot(accs,label="Training Performance")
    if test==True: 
        test_accs = np.array(picklin(path,'test_accs'))*100
        plt.axhline(test_accs[-1],linestyle='--',color='r',label="Test Performance")
    if val==True: 
        val_accs = np.array(picklin(path,'val_accs'))*100
        plt.plot(val_accs,label="Validation Performance")
    # print(test_accs)

    plt.title("Training Accuracy per Epoch", fontsize=20)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Classification Accuracy", fontsize=14)
    plt.ylim(0,100)
    

    plt.legend()
    plt.show()

dr = 'mnist_study'
exp = 'valtest_500'

dr = 'mnist_study'
exp = 'cifar_small_2'

path = os.path.join('../results',dr,exp)

plot_training(path,train=True,test=False,val=False)
# %%

string = "ra_neuron_to_dendrite_ri__beta_di_6.28319e+05__d_phi_r_0.2000__s_th_0.01__dt_0.10ns__ib_n_1.6024__ib_d_1.8524__working_master.soen"
print(len(string))
# %%
