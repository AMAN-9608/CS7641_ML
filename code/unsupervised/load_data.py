from sklearn.decomposition import PCA
import numpy as np
from PIL import Image
import os
import pickle

datapath = os.path.join(os.path.join("..", ".."), "data")
normalization_size = (400, 400)

def read_data(size = (400, 400)):
    # load training data
    training_path = os.path.join (datapath, 'train')
    pn_path = os.path.join(training_path, "PNEUMONIA")
    n_path = os.path.join(training_path, "NORMAL")
    filelist_p = [f for f in os.listdir(pn_path) if os.path.isfile(os.path.join(pn_path, f))]
    filelist_n = [f for f in os.listdir(n_path) if os.path.isfile(os.path.join(n_path, f))]
    
    images_pn =  np.array([np.array(Image.open(os.path.join(pn_path, fname)).resize(size)) for fname in filelist_p])
    images_n =  np.array([np.array(Image.open(os.path.join(n_path, fname)).resize(size)) for fname in filelist_n])

    images_pn = images_pn.reshape((-1, size[0]*size[1]))
    images_n = images_n.reshape((-1, size[0]*size[1]))

    # append labels
    images_n = np.concatenate((images_n, np.zeros((images_n.shape[0], 1))), axis=1)

    # pneumonia labels
    labels = np.array([1 if "bacteria" in f else 2 for f in filelist_p]).reshape((-1, 1))
    images_pn = np.concatenate((images_pn, labels), axis=1)

    total_data = np.concatenate((images_pn, images_n), axis=0)

    outfile = "data_{}_{}.pkl".format(size[0], size[1])
    file = open(os.path.join(datapath, outfile), 'wb')
    pickle.dump(total_data, file)

    return os.path.join(datapath, outfile)

    

if __name__ == "__main__":
    read_data(size=(2,2))