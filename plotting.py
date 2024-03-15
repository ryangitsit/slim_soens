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

def plot_nodes(
        nodes,
        title = None,
        ref   = True,
        flux  = True,
        ):

    plt.style.use('seaborn-v0_8-muted')
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig,ax = plt.subplots(len(nodes),1,figsize=(8,2*len(nodes)), sharex=True)
    for i,node in enumerate(nodes):
        if i == 0:
            ax[i].plot(node.dend_soma.signal,linewidth=4,color=colors[0],label="somatic signal")
            if flux==True: 
                ax[i].plot(node.dend_soma.flux,'--',linewidth=2,color=colors[1],label="somatic flux")
            if ref==True:  
                ax[i].plot(node.dend_ref.signal,':',linewidth=1,color=colors[2],label="refractory signal")
        else:
            ax[i].plot(node.dend_soma.signal,linewidth=4,color=colors[0])
            if flux==True: ax[i].plot(node.dend_soma.flux,'--',linewidth=2,color=colors[1])
            if ref==True:  ax[i].plot(node.dend_ref.signal,':',linewidth=1,color=colors[2])
        ax[i].set_title(node.name)
    
    if title is not None:
        fig.text(0.525, 0.95, title, ha='center',fontsize=18)
    fig.text(0.5, 0.05, 'Time (ns)', ha='center', fontsize=14)
    fig.text(0.05, 0.5, 'Unitless Signal and Flux', va='center', rotation='vertical', fontsize=14)
    fig.legend(bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.subplots_adjust(bottom=.125)
    plt.subplots_adjust(left=.125,right=.95,top=.9)
    plt.show()

def plot_letters(letters,letter=None):
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(letters),figsize=(8,6))
    for  j,(name,pixels) in enumerate(letters.items()):
        arrays = [[] for i in range(3)]
        count = 0
        for col in range(3):
            for row in range(3):
                arrays[col].append(pixels[count])
                count+=1
        pixels = np.array(arrays).reshape(3,3)

        axs[j].set_xticks([])
        axs[j].set_yticks([])
        axs[j].set_title(name,fontsize=14)

        if letter==name:
            axs[j].imshow(
                pixels,
                interpolation='nearest',
                cmap=cm.Oranges
                )
        else:
            axs[j].imshow(
                pixels,
                interpolation='nearest',
                cmap=cm.Blues
                )
    plt.show()