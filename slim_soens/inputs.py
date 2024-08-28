import numpy as np


def create_input():
    pass

def make_basic_convolutions():
    """
    Simple 9-pixel convolutions
    """
    k_vert = np.array([
        [0,1,0],
        [0,1,0],
        [0,1,0]])

    k_horz= np.array([
        [0,0,0],
        [1,1,1],
        [0,0,0]])

    k_up= np.array([
        [0,0,1],
        [0,1,0],
        [1,0,0]])

    k_down= np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1]])
    
    return [k_vert,k_horz,k_up,k_down]

def convolve_basic(sample,x,y,stride):
    """
    Returns feature maps of basic convolutions on some image
    """
    convolutions =  make_basic_convolutions()

    X = len(sample[0]) - x
    Y = len(sample) - y

    x_stride = y_stride = stride
    X = int(X/x_stride)
    Y = int(Y/y_stride)

    feature_maps = [np.zeros((X,Y)) for _ in range(len(convolutions))]

    for i in range(X):
        for j in range(Y):
            x1 = i*x_stride
            x2 = x1+x
            y1 = j*y_stride
            y2 = y1+y
            # print(f"{i,j} -> ({x1,x2}, {y1,y2})")
            window = sample[y1:y2,x1:x2]
            for f,feat in enumerate(feature_maps):
                feat[i][j] = (window*convolutions[f]).sum()

    return feature_maps

def features_to_input(feature_maps):
    flattened_features = []
    for i, feat in enumerate(feature_maps):
        flattened_features.append(feat.flatten())
    return np.concatenate(np.array(flattened_features))

def img_to_series(sample):
    return np.concatenate([np.ones(10)*val for val in sample.flatten()])[1000:7000]/255

def add_ssm_input(ssm,sample,B):
    for i,dend in enumerate(ssm.x):
        dend.input_flux = np.clip(img_to_series(sample)*B[i],-0.5,0.5)

def get_data(dataset,size,convolve=False,sequentialize=False,test_set=False):
    """
    Returns data and labels for appropriate dataset and size
    """
    if dataset == "mnist":
        from keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        if test_set==False:
            data = X_train[:size]
            labels = y_train[:size]
        else:
            print("Getting test data")
            data = X_test[:size]
            labels = y_test[:size]
    elif dataset == 'cifar':
        from keras.datasets import cifar10
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        print(X_train[0].shape, X_train[0].flatten().shape)
        data = X_train[:size]
        labels = y_train[:size]

    if convolve==True:
        if dataset == "mnist":
            x = 3
            y = 3
            stride = 1
            convolved_data = []
            for sample in data:
                feature_maps = convolve_basic(sample,x,y,stride)
                convolved_data.append(features_to_input(feature_maps))
            data = convolved_data

        elif dataset == 'cifar':
            x = 3
            y = 3
            stride = 1
            convolved_rgb_data = []
            for sample in data:
                feature_maps = np.concatenate(
                    [convolve_basic(sample[:,:,i],x,y,stride) for i in range(3)]
                    )
                convolved_rgb_data.append(features_to_input(feature_maps))
            
            data = convolved_rgb_data

    elif sequentialize==True:
        sequentialized_data = []
        for sample in data:
            sequentialized_data.append(np.concatenate(
                [np.ones(10)*val for val in sample.flatten()])[1000:7000]/255
            )
        data = sequentialized_data

    else:
        flattened_data = []
        for sample in data:
            flattened_data.append(sample.flatten())
        data = flattened_data

    return data, labels

def split_data(data, labels, tvt_split):
    size = len(labels)
    train_size = int(tvt_split[0]*size)
    val_size   = int(tvt_split[1]*size)
    train_data = data[:train_size]
    val_data   = data[train_size:train_size+val_size]
    test_data  = data[train_size+val_size:]

    train_labels = labels[:train_size]
    val_labels   = labels[train_size:train_size+val_size]
    test_labels  = labels[train_size+val_size:]

    return train_data,val_data,test_data,train_labels,val_labels,test_labels

