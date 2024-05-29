#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append('../')

from system_functions import *


#%%

def plot_training(path):
    plt.style.use('seaborn-v0_8-muted')
    plt.figure(figsize=(8,4))
    accs = np.array(picklin(path,'learning_accs'))*100
    test_accs = np.array(picklin(path,'test_accs'))*100
    val_accs = np.array(picklin(path,'val_accs'))*100
    print(test_accs)

    plt.title("Training Accuracy per Epoch", fontsize=20)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Classification Accuracy", fontsize=14)
    plt.plot(accs,label="Training Performance")
    plt.plot(val_accs,label="Validation Performance")
    plt.axhline(test_accs[-1],linestyle='--',color='r',label="Test Performance")
    plt.legend()
    plt.show()

dr = 'mnist_study'
exp = 'valtest'
path = os.path.join('../results',dr,exp)

plot_training(path)
# %%
