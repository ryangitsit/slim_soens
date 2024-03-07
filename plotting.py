import matplotlib.pyplot as plt
import numpy as np
def plot_adjacency(adjacency_matrix):
    for (j,i),label in np.ndenumerate(adjacency_matrix):
        if label!=0.0:plt.text(i,j,label,ha='center',va='center',fontsize=8)

    plt.imshow(adjacency_matrix)
    plt.show()