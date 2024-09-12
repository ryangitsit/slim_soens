import numpy as np

def weights_to_adjacency(weights):
    """
    Docstring
    """
    # collect total number of dendrites and add two for soma and refractory
    side_length = sum([len(group) for layer in weights for group in layer])+2 

    # initialize adjacency matrix
    adj_mat = np.zeros((side_length,side_length))
    adj_mat[1][0] = -0.85
    count = 2
    for l,layer in enumerate(weights):
        for g,group in enumerate(layer):
            for w,weight in enumerate(group):
                if l == 0:
                    outgoing_id = 0
                else:
                    leading_lyrs_len = sum(
                        [len(grp) for lay in weights[:l-1] for grp in lay]
                        )

                    outgoing_id = leading_lyrs_len + g + 2
            
                adj_mat[count][outgoing_id] = weight
                count+=1

    return adj_mat

def adjacency_to_weights(adj_mat):
    """
    Docstring
    """
    # weights = [[[]for _ in range(len(adj_mat))] for _ in range(len(adj_mat))]

    # for i,row in enumerate(adj_mat[2:]):
    #     for j,val in enumerate(row):
    #         if val!=0:
    #             weights[j][i].append(val)

    w = []
    for col_idx in range(len(adj_mat)):
        fan_ins = list(adj_mat[:,col_idx][adj_mat[:,col_idx]>0])
        if fan_ins != []:
            w.append(fan_ins)


    # w = []
    # for i,layer in enumerate(weights):
    #     for j,group in enumerate(layer):
    #         if group != []: w.append(layer)

    # w2 = []
    # for i,layer in enumerate(w):
    #     w2.append(np.concatenate(layer))

    return w



def get_name(obj):
    """
    Docstring
    """
    if hasattr(obj,'name'):
        return obj.name
    else:
        return None

def name_map(obj_list):
    """
    Docstring
    """
    return [*map(get_name,obj_list)]

def print_names(obj_list):
    """
    Docstring
    """
    print("\n")
    result = name_map(obj_list)
    print(*result, sep = "\n") 
    print("\n")
    return [*result]

def print_outgoing(dend):
    """
    Docstring
    """
    return (f"{dend.name} -> {name_map(dend.outgoing)}")

def outgoing_map(neuron):
    """
    Docstring
    """
    print("\n")
    result = map(print_outgoing,neuron.dendrite_list)
    print(*result, sep = "\n") 
    print("\n")
    return [*result]

def print_incoming(dend):
    """
    Docstring
    """
    return (f"{dend.name} <- {name_map(dend.incoming)}")

def incoming_map(neuron):
    """
    Docstring
    """
    print("\n")
    result = map(print_incoming,neuron.dendrite_list)
    print(*result, sep = "\n") 
    print("\n")
    return [*result]


def get_attr(obj,attr):
    """
    Docstring
    """
    if hasattr(obj,attr):
        return obj.__dict__[attr]
    else:
        return None

def attr_map(obj_list,attr):
    """
    Docstring
    """
    attrs = [attr for _ in range(len(obj_list))]
    return [*map(get_attr,obj_list,attrs)]


def print_attrs(obj_list,attrs):
    """
    Docstring
    """
    print("\n")
    results = []
    for attr in attrs:
        # print(attr)
        results.append(attr_map(obj_list,attr))

    for i,item in enumerate(results[0]):
        string = ""
        for result in results:
            string+=f"  {result[i]}  "
        print(string)
    print("\n")
    return [*result]

def get_jj_params():
    """
    Docstring
    """
    jj_params = {
        'c_j'       : 1.5e-13, 
        'r_j'       : 2.565564120477849, 
        'tau_0'     : 1.2827820602389245e-12, 
        'Ic'        : 9.999999999999999e-05, 
        'beta_c'    : 0.3, 
        'gamma'     : 1.5e-09, 
        'V_j'       : 0.0002565564120477849, 
        'omega_c'   : 779555647834.4772, 
        'omega_p'   : 1423267377164.988,
        't_tau_cnvt': 1e-9/1.2827820602389245e-12
        }
    return jj_params

def picklit(obj,path,name):
    import os
    import pickle
    try:
        os.makedirs(path)    
    except FileExistsError:
        pass
    pick = f'{path}/{name}.pickle'
    filehandler = open(pick, 'wb') 
    pickle.dump(obj, filehandler)
    filehandler.close()

def picklin(path,name):
    import os
    import pickle
    file = os.path.join(path, name)
    if '.pickle' in file:
        file = file
    else:
        file = file + '.pickle'
    # print(file)
    file_to_read = open(file, "rb")
    obj = pickle.load(file_to_read)
    file_to_read.close()
    return obj

def try_pickle(path,name,val,vals):
    """
    If saved array already exists
     - append val to it
    Otherwise 
     - create with vals
    """
    try:
        prev = picklin(path,name)
        prev.append(val)
        picklit(prev,path,name)
    except:
        picklit(vals,path,name)

def binary_fanin(layers):
    weights = [
        [np.random.rand(2) for _ in range(2**(l-1))] 
        for l in range(1,layers+1)
        ]
    return weights


def make_letters(patterns='zvn'):

    # non-noisy nine-pixel letters
    letters = {
        'z': [1,1,0,
              0,1,0,
              0,1,1],

        'v': [1,0,1,
              1,0,1,
              0,1,0],

        'n': [0,1,0,
              1,0,1,
              1,0,1]
    }

    if patterns == 'zvnx+':
        letters.update({
            'x': [1,0,1,
                  0,1,0,
                  1,0,1],
            '+': [0,1,0,
                  1,1,1,
                  0,1,0]
        })

    if patterns == 'all':
        letters.update({
            'x': [1,0,1,
                  0,1,0,
                  1,0,1],
            '+': [0,1,0,
                  1,1,1,
                  0,1,0],
            "|": 
                [0,1,0,
                 0,1,0,
                 0,1,0],
            "[": 
                [1,0,0,
                 1,0,0,
                 1,0,0,],
            "]": 
                [0,0,1,
                 0,0,1,
                 0,0,1,],
            "-": 
                [0,0,0,
                 1,1,1,
                 0,0,0],
            "_": 
                [0,0,0,
                 0,0,0,
                 1,1,1],
            "\\": 
                [1,0,0,
                 0,1,0,
                 0,0,1],
            "/": 
                [0,0,1,
                 0,1,0,
                 1,0,0],
            "*": 
                [1,1,1,
                 1,1,1,
                 1,1,1],
        })

    return letters


def make_inputs(letters,spike_time):

    # make the input spikes for different letters
    inputs = {}
    for name, pixels in letters.items():
        idx = np.where(np.array(letters[name])==1)[0]
        spike_times = np.ones(len(idx))*spike_time
        defined_spikes=[idx,spike_times]
        inputs[name]=defined_spikes
    return inputs

def make_repeated_inputs(letters,spike_time,repeats):

    # make the input spikes for different letters
    inputs = {}
    for name, pixels in letters.items():
        idx = np.where(np.array(letters[name])==1)[0]
        idx = np.concatenate([idx,idx+len(pixels)])
        spike_times = np.ones(len(idx))*spike_time
        defined_spikes=[idx,spike_times]
        inputs[name]=defined_spikes
    return inputs


def array_to_rows(array,channels):
    rows = [ [] for _ in range(channels) ]
    # print(len(array[0]),len(array[1]))
    for i,unit in enumerate(array[0]):
        # print(i,int(array[0][i]),int(array[1][i]),len(array[1]),len(array[1]))
        rows[int(unit)].append(array[1][i])
    return rows

def sizeup_obj(obj):
    import sys
    print(f"Size of {type(obj)} object = {sys.getsizeof(obj)}")
    for i,(k,v) in enumerate(obj.__dict__.items()):
        print(f"  {k}  -->  {sys.getsizeof(v)}")
    print("\n")

def clear_net(net):
    # import time
    # t1 = time.perf_counter()
    for node in net.nodes:
        node.dend_soma.spikes = []
        node.dend_soma.quiescence = 0
        for dend in node.dendrite_list:
            dend.signal = 0
            dend.flux   = 0

        for syn in node.synapse_list:
            syn.flux        = 0
            syn.spike_times = 0

        net.output_spikes = []
    del(net)
    # t2 = time.perf_counter()
    # print(f"Time to clear network = {np.round(t2-t1,5)}")

def check__net(net):
    print(dir(),"\n")
    sizeup_obj(net)
    sizeup_obj(net.nodes[0])
    sizeup_obj(net.nodes[0].dendrite_list[0])
    sizeup_obj(net.nodes[0].synapse_list[0])
    print("\n")

def no_ties(digit,spikes):
    sub = np.array(spikes) - spikes[digit] 
    if sum(n > 0 for n in sub) == 0 and sum(n == 0 for n in sub) == 1:
        return True
    else:
        return False
    
def count_total_elements(lst):
    total_elements = 0
 
    for item in lst:
        if isinstance(item, list) or isinstance(item, np.ndarray):
            total_elements += count_total_elements(item)
        else:
            total_elements += 1
 
    return total_elements

def mutual_inhibition(nodes,inhibition_strengh):
    from components import Synapse
    for i,node in enumerate(nodes):
        syn = Synapse(**{'dend_name':node.dend_soma.name}) 
        syn.outgoing.append(node.dend_soma)
        node.dend_soma.incoming.append((syn,inhibition_strengh))
        node.synapse_list.append(syn)

    for i,node_send in enumerate(nodes):
        for j,node_receive in enumerate(nodes):
            if i!=j:
                node_send.dend_soma.outgoing.append([node_receive.synapse_list[-1],1])
                node_receive.synapse_list[-1].incoming.append((node_send.dend_soma,1))
    
def print_dict(dct):
    print("\n")
    for i,(k,v) in enumerate(dct.items()):
        print(f" {i}   {k}  {v}")
    print("\n")

def rounded_percentage(num,total):
    return np.round(100*num/total,2)

def make_directed_connection(out_obj,in_obj,cs=1):
    """
    Make a connection from out-object to in-object.
        - Agnostic to type
    """
    out_obj.outgoing.append([in_obj,cs])
    
    in_obj.incoming.append([out_obj,cs])