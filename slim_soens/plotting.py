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
    plt.gca().invert_xaxis()
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
                ax[i].plot(
                    node.dend_soma.signal,linewidth=4,color=colors[0],label="somatic signal"
                    )
                if flux==True: 
                    ax[i].plot(
                        node.dend_soma.flux,'--',linewidth=2,color=colors[1],label="somatic flux"
                        )
                if ref==True:  
                    ax[i].plot(
                        node.dend_ref.signal,':',linewidth=1,color=colors[2],label="refractory signal"
                        )
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

            if len(node.dend_soma.spikes) > 0:
                ax[i].scatter(
                    np.array(node.dend_soma.spikes)-1,
                    np.ones(len(node.dend_soma.spikes))*node.dend_soma.threshold,
                    marker='x',color='k',zorder=100,s=50)

            ax[i].set_title(node.name)


        if title is not None:
            fig.text(0.525, 0.95, title, ha='center',fontsize=18)
        fig.text(0.5, 0.05, 'Time (ns)', ha='center', fontsize=14)
        fig.text(0.05, 0.5, 'Unitless Signal and Flux', va='center', rotation='vertical', fontsize=14)
        fig.legend(bbox_to_anchor=(1,.9))
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

        if len(node.dend_soma.spikes) > 0:
            plt.scatter(
                np.array(node.dend_soma.spikes)-1,
                np.ones(len(node.dend_soma.spikes))*node.dend_soma.threshold,
                marker='x',color='k',zorder=100,s=50)
        
        plt.title(node.name, fontsize=16)
        plt.subplots_adjust(bottom=.125)
        
        plt.xlabel('Time (ns)', fontsize=14)
        plt.ylabel('Unitless Signal and Flux',fontsize=14)
        plt.legend()

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

def plot_representations(nodes,shape=(28,28),disynaptic=False,activity=False):
    reps = []
    if disynaptic==True: 
        cols = 2
        fig,ax = plt.subplots(len(nodes),cols,figsize=(8,4*len(nodes)), sharex=True,sharey=True)
        for n,node in enumerate(nodes):
            
            learned_offsets_positive = []
            learned_offsets_negative = []

            for i,dend in enumerate(node.dendrite_list[2:]):
                if activity==True:
                    print("here")
                    val = dend.signal[-1]*dend.outgoing[0][1]
                else:
                    val = dend.flux_offset

                if dend.outgoing[0][1] >= 0:
                    learned_offsets_positive.append(val)
                else:
                    learned_offsets_negative.append(val)

            ax[n][0].imshow(np.array(learned_offsets_positive).reshape(shape),cmap="Greens")
            ax[n][1].imshow(np.array(learned_offsets_negative).reshape(shape),cmap="Reds")
            ax[n][0].set_xticks([])
            ax[n][1].set_yticks([])

        plt.show()
    else:
        cols = 1
        fig,ax = plt.subplots(len(nodes),cols,figsize=(8,4*len(nodes)), sharex=True,sharey=True)
        for n,node in enumerate(nodes):
            
            learned_offsets_positive = []
            learned_offsets_negative = []

            for i,dend in enumerate(node.dendrite_list[2:]):
                if activity==True:
                    val = dend.signal*dend.outgoing[0][1]
                else:
                    val = dend.flux_offset
                if dend.outgoing[0][1] >= 0:
                    learned_offsets_positive.append(val)
                else:
                    learned_offsets_negative.append(val)

            ax[n].imshow(np.array(learned_offsets_positive).reshape(shape),cmap="Greens")
            reps.append(np.array(learned_offsets_positive))
            ax[n].set_xticks([])
            ax[n].set_yticks([])

        plt.show()

    return reps

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


def structure(node):
    '''
    Plots arbitrary neuron structure
        - syntax
            -> SuperNode.plot_structure()
        - Weighting represented in line widths
        - Dashed lines inhibitory
        - Star is cell body
        - Dots are dendrites
    '''
    plt.style.use('seaborn-muted')
    # print(plt.__dict__['pcolor'].__doc__)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # arbor = node.dendrites
    weights = node.weights
    weights.insert(0,[[1]])
    arbor = weights
    print(weights)

    Ns = []
    for i,a in enumerate(node.weights):
        count = 0
        lsts = sum(type(el)== type([]) for el in a)
        if lsts > 0:
            for j in range(lsts):
                count+=len(node.weights[i][j])
        else: count = len(a)
        Ns.append(count)
    m=max(Ns)
    
    Y = [[] for i in range(len(weights))]
    G = []

    Ydot = []
    Xdot = []
    X_synapses = []
    Y_synapses = []
    syn_values = []
    dots = []
    x_ticks = []
    x_labels = []
    x_factor = 1
    y_factor = 5
    # colors = ['r','b','g']
    for i,layer in enumerate(weights):
        count=0
        groups = []
        for j,group in enumerate(layer):
            g = []
            for k,dend in enumerate(group):
                x = 1 + i*x_factor
                if i==0:
                    y = np.round(1+count*y_factor,2)
                    Y[i].append(y)
                elif i==len(arbor)-1:
                    y = np.round(np.mean(G[i-1]),2)
                    Y[i].append(y)
                else:
                    y = G[i-1][count]
                    Y[i].append(y)

                Xdot.append(x)
                Ydot.append(y)
                
                # syns = len(list(dend.synaptic_inputs))
                
                # if syns>0:
                #     if syns>1:
                #         y_space = np.arange(y-.5,y+.501,1/(syns-1))
                #     else:
                #         y_space = [y]
                #     for s,syn in enumerate(dend.synaptic_inputs):
                #         X_synapses.append(x-.1)
                #         Y_synapses.append(y_space[s])
                #         idx = list(dend.synaptic_connection_strengths.keys())[s]
                #         syn_values.append(dend.synaptic_connection_strengths[idx])
                
                if hasattr(dend, 'branch'):
                    branch = dend.branch
                else:
                    branch = None

                if hasattr(dend,'output_connection_strength'):
                    output = dend.output_connection_strength
                else:
                    output=None

                dot = [x,y,i,j,k,count,branch,output]
                dots.append(dot)
                g.append(y)
                count+=1
                x_ticks.append(x)
                x_labels.append(f"layer {len(weights)-(i+1)}")
            groups.append(np.mean(g))
        G.append(groups)
    plt.figure(figsize=(8,5))

    # labels = ['Basal','Proximal']
    count=0
    for i,dot1 in enumerate(dots):
        
        for ii,dot2 in enumerate(dots):
            if dot1[3] == dot2[5] and dot1[2] == dot2[2]-1:
                to_dot = dot2
        x1 = dot1[0]
        x2 = to_dot[0]
        y1 = dot1[1]
        y2 = to_dot[1]

        if dot1[6] != None:
            # color = color_map.colors[dot1[6]]
            color = colors[(dot1[6]+3)%len(colors)]
        else:
            color = 'k'
        if dot1[7] != None and dot1[7] != 0:
            width = np.max([int(np.abs(dot1[7]*5)),1])
        else:
            width = .01
        # print(i,dot1,'-->',to_dot)

        line_style = '-'
        if dot1[7] != None and dot1[7]<0:
            line_style='--'

        if to_dot[2]==len(arbor)-1 and to_dot!=dot1:
            # print("to soma")
            
            plt.plot(
                [x1,x2],[y1,y2],linestyle=line_style,
                color=color,linewidth=width,label=f'branch {dot1[6]}' #f"{labels[count]} Branch" 
                )
            count+=1
        else:
            plt.plot(
                [x1,x2],[y1,y2],
                linestyle=line_style,
                color=color,
                linewidth=width*1.5
                )
    
    if sum(Ns) > 30:
        ms = np.array([30,20,15,8])*15/sum(Ns)
        syn_values = np.array(syn_values)*8*15/sum(Ns)

    else:
        # ms = np.array([30,20,15,8])
        ms = np.array([30,20,11,11]) #NICEplot
        syn_values = np.array(syn_values)*200


    plt.plot(Xdot[-1],Ydot[-1],'*k',ms=ms[0])
    plt.plot(Xdot[-1],Ydot[-1],'*y',ms=ms[1],label='Soma')
    plt.plot(Xdot[0],Ydot[0],'ok',ms=ms[2],label='Dendrites')
    plt.plot(Xdot[1:-1],Ydot[1:-1],'ok',ms=ms[2])


    syn_colors = []
    for s in syn_values:
        if s > 0:
            syn_colors.append('r')
        else:
            syn_colors.append('b')
    syn_values = np.abs(syn_values)
    plt.scatter(
        X_synapses,
        Y_synapses,
        marker='>', 
        c=syn_colors,
        s=75,#syn_values,
        label='Synapses',
        )

    # plt.legend(borderpad=1)


    plt.legend(borderpad=1,markerscale=.7)

    x_labels[-1] += " (soma)"
    plt.xticks(x_ticks,x_labels,fontsize=14)
    plt.xlim(1-.1*len(arbor),len(arbor)*1.1)
    plt.ylim(1-y_factor,1+m*y_factor)
    plt.yticks([])
    plt.ylabel("Dendrites",fontsize=18)
    plt.xlabel("Layers",fontsize=18)
    plt.title("Dendritc Arbor",fontsize=20)
    plt.tight_layout()
    plt.show()


