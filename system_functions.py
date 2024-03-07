def weights_to_adjacency(weights):
    """
    Docstring
    """
    # collect total number of dendrites and add two for soma and refractory
    side_length = sum([len(group) for layer in weights for group in layer])+2 
    print(side_length)

    # initialize adjacency matrix
    adj_mat = np.zeros((side_length,side_length))

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

