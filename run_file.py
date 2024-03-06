from make_neuron import *
import timeit
import sys

weights_arbor = [
    [[0.2,0.5]],
    [[0.2,0.2],[0.5,0.5]],
    [[.21,.21],[.22,.22],[.51,.51],[.52,.52]]
]


# for i in range(50000):
# for i in range(1):

def create_neuron():
    return Neuron(weights=weights_arbor)

# timed = timeit.Timer(create_neuron).repeat(10, 100000)
# print(np.mean(timed),"\n")

neuron = create_neuron()

print(f"Neuron object size = {sys.getsizeof(neuron)}")


def print_outgoing(dend):
    if dend.outgoing is not None:
        return (f"{dend.name} -> {dend.outgoing}")
    else:
        return (f"{dend.name} -> {dend.outgoing}")

result = map(print_outgoing,neuron.dendrite_list)



print(*result, sep = "\n") 

"""
Try:
    weights -> adjacency -> make_dends -> connect_dends
"""