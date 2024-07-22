#%%
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('../')

from neuron import Neuron
from network import Network
import components

from weight_structures import *
from learning_rules import *
from plotting import *
from system_functions import *
from argparser import setup_argument_parser

#%%

from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_train = X_train[(y_train==0 )| (y_train ==1)]
# y_train = y_train[(y_train==0 )| (y_train ==1)]

#%%


d = 20
A = np.zeros((d,d))
for i in range(d):
    for j in range(d):
        coeff = (2*i+1)
        if i < j:
            A[i][j] = coeff*(-1)
        else:
            A[i][j] = coeff*((-1)**(i-j+1))

B = np.zeros((d,))
for i in range(d):
    B[i] = (2*i+1)*(-1)**i

A = A/25
B = B/25
print(f"A = \n{A}")
print(f"B = {B}")

# img = ax.imshow(A)
# ax.set_aspect("auto")
# plt.colorbar(img)
# plt.show()

#%%
from components import Soma, Dendrite
readout_layer = [Soma(neuron_name=f"readout_soma_{i}") for i in range(10)]

# for soma in readout_layer:
#     print(soma.name)

weights = [[B]]

arbor_params = [[[{'tau':5} for _ in range(len(B)) ]]]

ssm_neuron = Neuron(
    name='ssm_neuron',
    threshold = 0.1,
    weights=weights
    )

x = ssm_neuron.dendrite_list[2:]

ssm_neuron.normalize_fanin_symmetric()

for i in range(d):
    for j in range(d):
        # print(f"{x[i].name} -> {x[j].name} : {A[i][j]}")
        x[i].outgoing.append([x[j],A[i][j]])
        x[j].incoming.append([x[i],A[i][j]])


for j in range(len(readout_layer)):
    for i in range(len(B)):
        # print(f"{x[i].name} -> {readout_layer[j].name} : {A[i][j]}")
        x[i].outgoing.append([readout_layer[j],.1])
        readout_layer[j].incoming.append([x[i],.1])

    ssm_neuron.dendrite_list.append(readout_layer[j])

T = 784 #141
runs = 1000
samples = 1000 #4998
train_split = 0.8
train_bound = int(train_split*samples)
plot=True
runs = 10000
samples = 100
classes = 10

mult_trajs = [[] for _ in range(d*classes)]
total_spiking = 0
for run in range(runs):
    success = 0
    test_success = 0
    for sample in range(samples):

        if sample < train_bound:
            mode = "train"
        else:
            mode = "test"

        u     = X_train[sample].reshape(784,)/(256*2)
        label = y_train[sample]


        for i,dend in enumerate(x):
            x[i].flux_offset = u[:T]*B[i]

        net = Network(
            run_simulation = True,
            nodes          = [ssm_neuron],
            duration       = T,
        )

        targets = np.zeros(classes)
        targets[label] = 0.72
        errors = []
        readout_signals = []
        for c in range(classes):
            spikes = len(readout_layer[c].spikes)
            if spikes > 0:
                total_spiking += 1
                # plot_nodes([ssm_neuron],dendrites=True)
            out_signal = np.mean(readout_layer[c].signal)
            readout_signals.append(out_signal)
            error = targets[c] - out_signal
            errors.append(error)

        # print(f"Digit {label} -> {np.argmax(readout_signals)} -- {readout_signals} -- {errors}  -- {np.argmax(targets)==np.argmax(readout_signals)}")

        if sample==0 and run%100==0: 
            # plot_nodes([ssm_neuron],dendrites=True)
            plt.title(f"Digit {label} -> {np.argmax(readout_signals)}")
            for r,read in enumerate(readout_layer):
                # plt.plot(read.flux,'--',label=f"f{r}")
                plt.plot(read.signal,label=f"s{r}")
            plt.legend()
            plt.show()

            # print(targets[c],error)
            
         
        if mode=="train":
            if np.argmax(targets)==np.argmax(readout_signals): success +=1
            count=0
            for out in range(classes):
                for i,dend in enumerate(x):

                    eta = 0.005 #/(1+run)
                    update = (
                        errors[out]*eta
                        *np.mean(dend.signal)
                        # *np.sign(readout_layer[out].incoming[i][1])
                        )
                    
                    readout_layer[out].incoming[i][1] += update

                    mult_trajs[count].append(readout_layer[out].incoming[i][1])
                    count+=1

        elif mode=="test":
            if np.argmax(targets)==np.argmax(readout_signals):  test_success +=1

        clear_net(net)
    train_acc = np.round(success/train_bound,2)
    test_acc = np.round(test_success/(4998-train_bound),2)
    print(f"Epoch {run} performance => train : {train_acc} :: test : {test_acc}")
#%%
print(total_spiking)
for xi in mult_trajs:
    plt.plot(xi)
plt.show()

# plot_nodes([ssm_neuron],dendrites=True)
# plot_by_layer(ssm_neuron,layers=2)

#%%
print(len(mult_trajs))
# for i,dend in enumerate(x):

# for i in range(d):
#     dend = x[i]
#     print(dend.name)
#     for obj,strengths in dend.outgoing:
#         print("  ",obj.name,strengths)

print(x[0].outgoing)
for i in range(11):
    print(ssm_neuron.dend_soma.incoming[i][0].name,ssm_neuron.dend_soma.incoming[i][1])
print("\n")
for i,dend in enumerate(x):
    print(dend.name,dend.outgoing[0][1])

#%%
print(len(labels))
print(sum(labels))
plt.plot(labels)

#%%

decay = [0.001/(1+2*_) for _ in range(10)]
plt.plot(decay)
plt.show()

arr = np.arange(10)
np.random.shuffle(arr)

arr2 = np.arange(10)*2
arr2 = arr2[arr]
print(arr2)
# %%
