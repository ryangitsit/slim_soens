import numpy as np

def make_rand_weights():
    W = [
    [np.random.rand(3)],
    np.random.rand(3,3)
    ]
    return W

def make_uniform_weights():
    W = [
    [np.ones((3,))],
    np.ones((3,3))
    ]
    return W

def make_hifan_weights():
    W = [
    [np.ones((9,))],
    ]
    return W

def make_doubled_weights():
    W = [
    [np.ones((3,))],
    np.ones((3,3)),
    [[-1,1] for _ in range(9)]
    ]
    arbor_params = [
        [[{'update_type':'normal'} if weight>0 else {'update':'inverted'}
            for w,weight in enumerate(group)] for g,group in enumerate(layer)
        ]
        for l,layer in enumerate(W)]
    return W


def make_extended_doubled_weights():
    W = [
    [np.ones((3,))],
    np.ones((3,6)),
    [[-1,1] for _ in range(18)]
    ]

    return W

def make_symmetric_weights(p_neg=.2):
    W = [
    [np.random.rand(3)],
    # [[0.3*np.random.choice([-1,1], p=[p_neg,1-p_neg], size=1)[0] for _ in range(3)]],
    [[0.3*np.random.choice([-1,1], p=[p_neg,1-p_neg], size=1)[0] for _ in range(3)] for _ in range(3)]
    ]
    print(W)
    return W

def make_crafted_weights(letter,pixels,symmetry=False):
    count = 0
    synaptice_layer = []
    for i in range(3):
        group = []
        for j in range(3):
            if    symmetry==False:  w=pixels[count]
            elif  pixels[count]==0: w=-1
            else: w=1
            group.append(w)
            count+=1
        synaptice_layer.append(group)

    W = [
    [np.random.rand(3)],
    synaptice_layer
    ]
    return W

def make_double_tree():
    W = [
    [np.ones((3,))],
    np.ones((3,6))
    ]
    return W

def make_hybrid_weights(letter,pixels,symmetry=False):
    count = 0
    synaptice_layer = []
    for i in range(3):
        group = []
        for j in range(3):
            if    symmetry==False:  w=pixels[count]*0.3
            elif  pixels[count]==0: w=-1*0.3
            else: w=0.3
            group.append(w)
            count+=1
        synaptice_layer.append(group)

    W = [
    [np.random.rand(2)],
    np.random.rand(2,3),
    np.concatenate([synaptice_layer,np.random.rand(3,3)])
    ]

    arbor_params = [
        [[{'update':True} if weight!=np.abs(0.3) else {'update':False}
            for w,weight in enumerate(group)] for g,group in enumerate(layer)
        ]
        for l,layer in enumerate(W)]
    return W, arbor_params

def make_random_shallow_weights(shape):
    return [[np.random.rand(shape)]]

def make_disynaptic_weights(shape):
    return [[[-1,1] for _ in range(shape)]]

def make_random_deep_weights(shape):
    print("Making deep trees")
    weights = []
    layers = int(np.log2(shape)/np.log2(2))
    grp = 2
    for l in range(layers):
        groups = int(shape/(2**(l))/2)
        layer = []
        if l != 0:
            grp = int(len(weights[-1])/groups)
        for group in range(groups):
            layer.append(np.random.rand(grp))
        weights.append(layer)
    weights.reverse()
    return weights