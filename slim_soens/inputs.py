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
    intype = 'steady'
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

        datashape  = data[0].shape
        inputshape = 784

    if dataset == "fashion_mnist":
        from keras.datasets import fashion_mnist
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        if test_set==False:
            data = X_train[:size]
            labels = y_train[:size]
        else:
            print("Getting test data")
            data = X_test[:size]
            labels = y_test[:size]

        print(len(X_train),len(X_test))
        datashape  = data[0].shape
        inputshape = data[0].shape[0]*data[1].shape[1]
        

    elif dataset == 'cifar':
        from keras.datasets import cifar10
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        print(X_train[0].shape, X_train[0].flatten().shape)
        data = X_train[:size]
        labels = y_train[:size]
        datashape  = data[0].shape
        inputshape = datashape[0]

    elif dataset == 'heidelberg':
        import os
        import urllib.request
        import gzip, shutil
        from tensorflow.keras.utils import get_file
        import tables
        cache_dir=os.path.expanduser("~/data")
        cache_subdir="hdspikes"
        print("Using cache dir: %s"%cache_dir)
        # The remote directory with the data files
        base_url = "https://zenkelab.org/datasets"
        # Retrieve MD5 hashes from remote
        response = urllib.request.urlopen("%s/md5sums.txt"%base_url)
        data = response.read() 
        lines = data.decode('utf-8').split("\n")
        file_hashes = { line.split()[1]:line.split()[0] for line in lines if len(line.split())==2 }
        def get_and_gunzip(origin, filename, md5hash=None):
            gz_file_path = get_file(filename, origin, md5_hash=md5hash, cache_dir=cache_dir, cache_subdir=cache_subdir)
            hdf5_file_path=gz_file_path[:-3]
            if not os.path.isfile(hdf5_file_path) or os.path.getctime(gz_file_path) > os.path.getctime(hdf5_file_path):
                print("Decompressing %s"%gz_file_path)
                with gzip.open(gz_file_path, 'r') as f_in, open(hdf5_file_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return hdf5_file_path
        # Download the Spiking Heidelberg Digits (SHD) dataset
        files = [ "shd_train.h5.gz", 
                "shd_test.h5.gz",
                ]
        for fn in files:
            origin = "%s/%s"%(base_url,fn)
            hdf5_file_path = get_and_gunzip(origin, fn, md5hash=file_hashes[fn])
            print(hdf5_file_path)
        fileh = tables.open_file(hdf5_file_path, mode='r')
        units = fileh.root.spikes.units[:size]
        times = fileh.root.spikes.times[:size]
        labels = fileh.root.labels[:size]
        data = []
        for i in range(size):
            data.append([units[i],times[i]])

        inputshape = 700
        intype = 'spikes'

    elif dataset == 'mnist_preconvolved':
        data   = np.load("../datasets/preconv/MNIST_train_X_conv.npy")
        labels = np.load("../datasets/preconv/MNIST_train_y_conv.npy")
        inputshape = len(data[0])
        print(inputshape, labels[:10])

    elif dataset == 'mnist_preconvolved_ext':
        data   = np.load("../datasets/preconv/MNIST_train_X_conv_ext.npy")
        labels = np.load("../datasets/preconv/MNIST_train_y_conv_ext.npy")
        inputshape = len(data[0])
        print(inputshape, labels[:10])

    elif dataset == "fashion_preconvolved_ext":
        data   = np.load("../datasets/preconv/fashionMNIST_train_X_conv_ext.npy")
        labels = np.load("../datasets/preconv/fashionMNIST_train_y_conv_ext.npy")
        inputshape = len(data[0])
        print(inputshape, labels[:10])

    elif dataset == "fashion_preconvolved_ext10":
        data   = np.load("../datasets/preconv/fashionMNIST_train_X_conv_ext10.npy")
        labels = np.load("../datasets/preconv/fashionMNIST_train_y_conv_ext10.npy")
        inputshape = len(data[0])
        print(inputshape, labels[:10])
    

    if convolve==True:
        if dataset == "mnist" or dataset == "fashion_mnist":
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

        elif dataset == 'heidelberg':
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

    elif intype=='steady':
        flattened_data = []
        for sample in data:
            flattened_data.append(sample.flatten())
        data = flattened_data

    return data, labels, inputshape, intype

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

def get_ECG5000():
    import pandas as pd

    dataframe = pd.read_csv(
        'http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', 
        header=None
        )

    raw_data = dataframe.values
    dataframe.head()
    labels = raw_data[:, -1]
    data = raw_data[:, 0:-1]

    shuff = np.arange(len(labels))
    np.random.shuffle(shuff)
    data   = data[shuff]
    labels = labels[shuff]

    # plt.title(labels[0])
    # plt.plot(data[0])
    # plt.show()

    return data, labels

# data, labels = get_ECG5000()
