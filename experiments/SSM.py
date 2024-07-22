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

u = np.loadtxt('../data/mackey.txt')
print(f"Mackey-Glass Chaotic Timeseries loaded of length {len(u)}")
plt.plot(u[:500])
plt.show()

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

fig, ax = plt.subplots()
img = ax.imshow(A)
ax.set_aspect("auto")
plt.colorbar(img)
plt.show()

#%%


weights = [
    [np.ones(d)],
    [[_] for _ in B]
    ]

ssm_neuron = Neuron(
    name='ssm_neuron',
    threshold = 0.25,
    weights=weights
    )

x = ssm_neuron.dendrite_list[d+2:]

y_pre = ssm_neuron.dendrite_list[2:d+2]
# print_attrs(ssm_neuron.dendrite_list,['name'])

# print("x")
# print_attrs(x,['name'])

# print("y")
# print_attrs(y_pre,['name'])

ssm_neuron.normalize_fanin_symmetric()

for i in range(d):
    for j in range(d):
        x[i].outgoing.append([x[j],A[i][j]])
        x[j].incoming.append([x[i],A[i][j]])

T = 100


mult_trajs = [[] for _ in range(d)]
runs = 1000
eta = 0.01
for run in range(runs):
    for t in range(T):
        for i,dend in enumerate(x):
            x[i].flux_offset = u[t:t+2]*B[i]

        net = Network(
            run_simulation = True,
            nodes          = [ssm_neuron],
            duration       = 2,
        )
        last_flux = ssm_neuron.dend_soma.flux[-1]
        error = u[t+1] - last_flux
        # print(error)
        
        for i,dend in enumerate(y_pre):
            # update = error*eta #*np.mean(dend.signal)
            # print(update)
            # dend.flux_offset += update
            
            eta = 0.001 #/(1+0.5*run)
            # update = error*eta*np.mean(dend.flux)*np.sign(ssm_neuron.dend_soma.incoming[i+1][1])
            update = np.mean(dend.flux)*error*eta
            ssm_neuron.dend_soma.incoming[i+1][1] += update
            mult_trajs[i].append(ssm_neuron.dend_soma.incoming[i+1][1])

        clear_net(net)


for i,dend in enumerate(x):
    x[i].flux_offset = u[:T]*B[i]

net = Network(
    run_simulation = True,
    nodes          = [ssm_neuron],
    duration       = T,
)
# plot_nodes([ssm_neuron],dendrites=True)
# plot_by_layer(ssm_neuron,layers=3)

plt.figure(figsize=(8,4))
plt.title("Untrained Signal Processing",fontsize=18)
plt.plot(u[:500]*0.05,label="u(t)")
plt.plot(ssm_neuron.dend_soma.flux[:500],linewidth=2,label="y(t)")
plt.xlabel("Time (ns)",fontsize=14)
plt.ylabel("Received Flux",fontsize=14)
plt.legend()
plt.show()

plt.figure(figsize=(8,2))
for xi in mult_trajs:
    plt.plot(xi)
plt.show()


#%%
weights = [[B]]

arbor_params = [[[{'tau':5} for _ in range(len(B)) ]]]

ssm_neuron = Neuron(
    name='ssm_neuron',
    threshold = 0.25,
    weights=weights,
    arbor_params=arbor_params
    )

x = ssm_neuron.dendrite_list[2:]

ssm_neuron.normalize_fanin_symmetric()

for i in range(d):
    for j in range(d):
        x[i].outgoing.append([x[j],A[i][j]])
        x[j].incoming.append([x[i],A[i][j]])

T = 500
mult_trajs = [[] for _ in range(d)]
runs = 10

# u = u[25:]

# for run in range(runs):
#     errors = []
#     for t in range(T):
#         for i,dend in enumerate(x):
#             x[i].flux_offset = u[t:t+2]*B[i]

#         net = Network(
#             run_simulation = True,
#             nodes          = [ssm_neuron],
#             duration       = 2,
#         )
#         error = u[t+1] - ssm_neuron.dend_soma.flux[-1]
#         errors.append(error)
#         for i,dend in enumerate(x):
#             eta = 0.0001/(1+.5*run)

#             update = error*eta*np.mean(dend.flux)*np.sign(ssm_neuron.dend_soma.incoming[i+1][1])
#             # update = np.mean(dend.flux)*error*eta
#             ssm_neuron.dend_soma.incoming[i+1][1] += update
#             mult_trajs[i].append(ssm_neuron.dend_soma.incoming[i+1][1])

        # clear_net(net)

# plt.plot(errors)
# plt.show()

for i,dend in enumerate(x):
    x[i].flux_offset = u[:T]*B[i]

net = Network(
    run_simulation = True,
    nodes          = [ssm_neuron],
    duration       = T,
)


plt.figure(figsize=(8,4))
plt.title("Untrained Signal Processing",fontsize=18)
plt.plot(u[:500]*0.05,label="u(t)")
plt.plot(ssm_neuron.dend_soma.flux[:500],linewidth=2,label="y(t)")
plt.xlabel("Time (ns)",fontsize=14)
plt.ylabel("Received Flux",fontsize=14)
plt.legend()
plt.show()

plt.figure(figsize=(8,2))
for xi in mult_trajs:
    plt.plot(xi)
plt.show()

plot_nodes(ssm_neuron,dendrites=True)


#%%
import pandas as pd

dataframe = pd.read_csv(
    'http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', 
    header=None
    )

raw_data = dataframe.values
dataframe.head()

# The last element contains the labels
labels = raw_data[:, -1]

# The other data points are the electrocadriogram data
data = raw_data[:, 0:-1]

shuff = np.arange(len(labels))
np.random.shuffle(shuff)

data   = data[shuff]
labels = labels[shuff]

plt.title(labels[0])
plt.plot(data[0])
plt.show()


#%%
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train[(y_train==0 )| (y_train ==1)]
y_train = y_train[(y_train==0 )| (y_train ==1)]

#%%

weights = [[B]]
arbor_params = [[[{'tau':5} for _ in range(len(B)) ]]]

ssm_neuron = Neuron(
    name='ssm_neuron',
    threshold = 0.25,
    weights=weights
    )

x = ssm_neuron.dendrite_list[2:]

ssm_neuron.normalize_fanin_symmetric()

for i in range(d):
    for j in range(d):
        # print(f"{x[i].name} -> {x[j].name} : {A[i][j]}")
        x[i].outgoing.append([x[j],A[i][j]])
        x[j].incoming.append([x[i],A[i][j]])


T = 784 #141
runs = 1000
samples = 1000 #4998
mult_trajs = [[] for _ in range(d)]
train_split = 0.8
train_bound = int(train_split*samples)
plot=True


#%%

for run in range(runs):
    success = 0
    test_success = 0
    for sample in range(samples):

        if sample < train_bound:
            mode = "train"
        else:
            mode = "test"

        # u     = data[sample]
        # label = labels[sample]
        
        u     = X_train[sample].reshape(784,)/(256*2)
        label = y_train[sample]
        if label == 0: label =1
        else: label = 0


        for i,dend in enumerate(x):
            x[i].flux_offset = u[:T]*B[i]

        net = Network(
            run_simulation = True,
            nodes          = [ssm_neuron],
            duration       = T,
        )
        # plot_nodes([ssm_neuron])
        # if run==1 and label==0 and plot==True and sample>4000:
        #     plot_nodes([ssm_neuron],dendrites=True)
        #     print(f"{label} -> {len(ssm_neuron.dend_soma.spikes)}")
        #     plot=False
        error = label - len(ssm_neuron.dend_soma.spikes)
         
        if mode=="train":
            if error == 0: success +=1
            for i,dend in enumerate(x):
                # print(dend.outgoing[0][1])
                eta = 0.001 #/(1+run)
                update = error*eta*np.mean(dend.signal) #*error*.0001
                # update = update = error*eta*np.mean(dend.flux)*np.sign(ssm_neuron.dend_soma.incoming[i+1][1])

                # print(update,np.mean(dend.signal),error)
                # print(dend.name,dend.outgoing[0][1],update, " --> ",dend.outgoing[0][1]*update)

                # dend.outgoing[0][1] += update
                ssm_neuron.dend_soma.incoming[i+1][1] += update

                mult_trajs[i].append(ssm_neuron.dend_soma.incoming[i+1][1])
                # print(dend.outgoing[0][1])

        elif mode=="test":
            if error == 0:  test_success +=1

        clear_net(net)
    train_acc = np.round(success/train_bound,2)
    test_acc = np.round(test_success/(4998-train_bound),2)
    print(f"Epoch {run} performance => train : {train_acc} :: test : {test_acc}")

#%%
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
