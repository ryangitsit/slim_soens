import numpy as np

def weights_to_adjacency(weights):
    """
    Docstring
    """
    # collect total number of dendrites and add two for soma and refractory
    side_length = sum([len(group) for layer in weights for group in layer])+2 
    print(side_length)

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
    if hasattr(obj,'name'):
        return obj.name
    else:
        return None

def name_map(obj_list):
    return [*map(get_name,obj_list)]

def print_outgoing(dend):
    return (f"{dend.name} -> {name_map(dend.outgoing)}")

def outgoing_map(neuron):
    print("\n")
    result = map(print_outgoing,neuron.dendrite_list)
    print(*result, sep = "\n") 
    print("\n")
    return [*result]

def print_incoming(dend):
    return (f"{dend.name} <- {name_map(dend.incoming)}")

def incoming_map(neuron):
    print("\n")
    result = map(print_incoming,neuron.dendrite_list)
    print(*result, sep = "\n") 
    print("\n")
    return [*result]

