import numpy as np

def weights_to_adjacency(weights):
    """
    Docstring
    """
    # collect total number of dendrites and add two for soma and refractory
    side_length = sum([len(group) for layer in weights for group in layer])+2 

    # initialize adjacency matrix
    adj_mat = np.zeros((side_length,side_length))
    adj_mat[1][0] = 0.5
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
    return obj.__dict__[attr]

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
        print(attr)
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
            "|  ": 
                [1,0,0,
                 1,0,0,
                 1,0,0,],
            "  |": 
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
            "[]": 
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


def array_to_rows(array,channels):
    rows = [ [] for _ in range(channels) ]
    for i in range(len(array[0])):
        rows[int(array[0][i])].append(array[1][i])
    return rows

