import matplotlib.pyplot as plt
import numpy as np
def heatmap_adjacency(adjacency_matrix):
    for (j,i),label in np.ndenumerate(adjacency_matrix):
        if label!=0.0:plt.text(i,j,label,ha='center',va='center',fontsize=8)

    plt.imshow(adjacency_matrix)
    plt.show()

def graph_adjacency(W,dims):
    import networkx as nx 
    plt.figure(figsize=(8,6))
    G = nx.DiGraph() 
    count = 1
    x = 0
    for i,row in enumerate(W): 
        dim = dims[x-1]
        # print(f"Node {i} | dimidx {x} | dim {dim} | sum {sum(dims[:x])} | count {count}")

        G.add_node(i,pos=(x+1,count+len(W)/2 - (8+x+1)))
        if count >= dims[x]:
            x+=1
            count=0
        count+=1

    for i,row in enumerate(W): 
        for j,val in enumerate(row): 
            if val > 0: G.add_edge(i,j)
            
    pos=nx.get_node_attributes(G,'pos')
    nx.draw(G,pos,with_labels = True,node_size = 250) 
    plt.show() 
    del(G)
