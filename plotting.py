import matplotlib.pyplot as plt
import numpy as np
import components

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
            if val != 0: G.add_edge(i,j, weight=val)
            
    epositive = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0]
    enegative = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] < 0] 
    labels = {}    
    for node in G.nodes():
        labels[node] = node
    pos=nx.get_node_attributes(G,'pos')
    # nx.draw(G,pos,with_labels = True,node_size = 250) 
    nx.draw_networkx_nodes(G, pos, node_size=250)
    nx.draw_networkx_labels(G,pos,labels,font_size=10,font_color='black')
    nx.draw_networkx_edges(
        G, pos, edgelist=epositive, width=2, alpha=0.5, edge_color="black", 
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=enegative, width=2, alpha=0.5, edge_color="black", style="dashed"
    )
    plt.show() 
    del(G)

def plot_nodes(
        nodes,
        title     = None,
        ref       = True,
        flux      = True,
        dendrites = False,
        weighting = False,
        ):

    plt.style.use('seaborn-v0_8-muted')
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if type(nodes)==list and len(nodes)>1:
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

            # if dendrites==True:
            #     for d,dend in enumerate(node.dendrite_list[2:]):
            #         ax[i].plot(dend.flux,'--',linewidth=1)

            if dendrites==True:
                w = 1
                for d,dend in enumerate(node.dendrite_list[2:]):
                    if weighting==True: w = dend.outgoing[0][1]
                    ax[i].plot(dend.signal*w,'--',linewidth=1)

            ax[i].set_title(node.name)


        if title is not None:
            fig.text(0.525, 0.95, title, ha='center',fontsize=18)
        fig.text(0.5, 0.05, 'Time (ns)', ha='center', fontsize=14)
        fig.text(0.05, 0.5, 'Unitless Signal and Flux', va='center', rotation='vertical', fontsize=14)
        fig.legend(bbox_to_anchor=(1,1))
        plt.tight_layout()
        plt.subplots_adjust(bottom=.125)
        plt.subplots_adjust(left=.125,right=.95,top=.9)
    else:
        if type(nodes)==list: node = nodes[0]
        else: node = nodes
        plt.figure(figsize=(8,4))
        plt.plot(node.dend_soma.signal,linewidth=4,color=colors[0],label="somatic signal")
        if flux==True: 
            plt.plot(node.dend_soma.flux,'--',linewidth=2,color=colors[1],label="somatic flux")
        if ref==True:  
            plt.plot(node.dend_ref.signal,':',linewidth=1,color=colors[2],label="refractory signal")
        if dendrites==True:
            w = 1
            for d,dend in enumerate(node.dendrite_list[2:]):
                if weighting==True: w = dend.outgoing[0][1]
                plt.plot(dend.signal*w,'--',linewidth=1)
        plt.title(node.name, fontsize=16)
        plt.subplots_adjust(bottom=.125)
        
        plt.xlabel('Time (ns)', fontsize=14)
        plt.ylabel('Unitless Signal and Flux',fontsize=14)

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

def plot_synapse_inversions(nodes,title="Double-Synapse Flux Offsets",pattern_idx=None):
    fig,ax = plt.subplots(len(nodes),1,figsize=(6,2.25*len(nodes)),sharex=True)
    for n,node in enumerate(nodes):
        neg_syns = []
        pos_syns = []

        active_pos = []
        inactive_pos = []

        active_neg = []
        inactive_neg= []

        active_syns=0
        for i,dend in enumerate(node.dendrite_list):
            if dend.loc[0]==3:

                if np.mean(dend.signal) > 0: 
                    active_syns +=1
                    if dend.outgoing[0][1] < 0:
                        neg_syns.append(dend.flux_offset)
                        # plt.plot(dend.signal,'--')
                        active_neg.append(dend.flux_offset)

                    else:
                        active_pos.append(dend.flux_offset)
                        pos_syns.append(dend.flux_offset)
                        # plt.plot(dend.signal)
                else:
                    if dend.outgoing[0][1] < 0:
                        neg_syns.append(dend.flux_offset)
                        inactive_neg.append(dend.flux_offset)

                    else:
                        pos_syns.append(dend.flux_offset)
                        inactive_pos.append(dend.flux_offset)


        ax[n].hist(active_neg,color='r'  ,bins=30)
        ax[n].hist(active_pos,color='g'  ,bins=30)

        ax[n].hist(inactive_neg,color='r',bins=30,alpha=0.3)
        ax[n].hist(inactive_pos,color='g',bins=30,alpha=0.3)
        ax[n].set_title(node.name,y=.75,x=.3)
        
        # plt.hist(neg_syns,color='r',bins=50,alpha=0.3)
        # plt.hist(pos_syns,color='g',bins=50,alpha=0.3)
    if pattern_idx is not None: ax[pattern_idx].set_facecolor('lightgrey')
    plt.suptitle(title)
    plt.show()

def plot_trajectories(nodes,double_dends=False):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig,ax = plt.subplots(len(nodes),1,figsize=(8,2.25*len(nodes)), sharex=True)
    for n,node in enumerate(nodes):

        ax[n].set_title(f"Update Trajectory of {node.name} Arbor",fontsize=12)
        for dend in node.dendrite_list:
            if hasattr(dend,'update_traj') and 'ref' not in dend.name:
                if isinstance(dend,components.Soma): 
                    lw = 4
                    c = colors[0]
                    line='solid'
                elif dend.loc[0]==1:
                    c = colors[3] 
                    lw = 2 
                    line = 'dashed'
                elif dend.loc[0]==4:
                    if dend.outgoing[0][1] < 0:
                        c = colors[1] 
                    else:
                        c = colors[5]
                    lw = 2   
                    line = 'dotted'
                else:
                    c = colors[2]
                    lw =0.5
                    line = 'dotted'

                ax[n].plot(np.array(dend.update_traj),linewidth=lw,linestyle=line,label=dend.name)

        # plt.legend(bbox_to_anchor=(1.01,1))
        # ax[n].set_x_label("Updates",fontsize=14)
        # ax[n].set_y_label("Flux Offset",fontsize=14)
    fig.tight_layout()
    plt.show()

def plot_by_layer(node,layers,flux=False):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plt.figure(figsize=(8,4))
    layers_encountered = []
    for dend in node.dendrite_list[:1] + node.dendrite_list[2:]:
        layer = dend.loc[0]
        
        lw = layers - layer
        c=colors[layer%len(colors)]
        line = "solid"
        if dend.loc[0] != 0 and dend.outgoing[0][1]<0: line = "dashed"

        if layer not in set(layers_encountered):
            plt.plot(dend.signal,linewidth=lw,linestyle=line,label=f"Layer {layer}", color=c)
        else:
            plt.plot(dend.signal,linewidth=lw,linestyle=line,color=c)
            if flux == True:
                plt.plot(dend.flux,'--',linewidth=lw,color=c)

        layers_encountered.append(layer)
    plt.legend()
    plt.title(node.name)
    plt.show()

def plot_representations(nodes,shape=(28,28)):

    fig,ax = plt.subplots(len(nodes),2,figsize=(8,4*len(nodes)), sharex=True,sharey=True)
    for n,node in enumerate(nodes):
        
        learned_offsets_positive = []
        learned_offsets_negative = []

        for i,dend in enumerate(node.dendrite_list[2:]):
            if dend.outgoing[0][1] >= 0:
                learned_offsets_positive.append(dend.flux_offset)
            else:
                learned_offsets_negative.append(dend.flux_offset)

        ax[n][0].imshow(np.array(learned_offsets_positive).reshape(shape),cmap="Greens")
        ax[n][1].imshow(np.array(learned_offsets_negative).reshape(shape),cmap="Reds")
        ax[n][0].set_xticks([])
        ax[n][1].set_yticks([])

    plt.show()

def plot_res_spikes(digits,samples,res_spikes):
    fig,ax = plt.subplots(digits,samples,figsize=(18,12), sharex=True,sharey=True)
    for i in range(digits):
        for j in range(samples):
            print(i,j,end="\r")
            inpt = res_spikes[i][j]
            # print(inpt)
            ax[i][j].plot(inpt[1],inpt[0],'.k',ms=0.25)
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
    # plt.title(f"RNN MNIST Spikes")
    plt.show()