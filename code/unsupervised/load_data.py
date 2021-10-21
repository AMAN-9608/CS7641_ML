from sklearn.decomposition import PCA
import numpy as np
from PIL import Image
import os
import tqdm

datapath = os.path.join(os.path.join("..", ".."), "data")

def load_data(size = (400, 400)):
    # load training data
    training_path = os.path.join (datapath, 'train')
    pn_path = os.path.join(training_path, "PNEUMONIA")
    n_path = os.path.join(training_path, "NORMAL")
    filelist_p = [f for f in os.listdir(pn_path) if os.path.isfile(os.path.join(pn_path, f))]
    filelist_n = [f for f in os.listdir(n_path) if os.path.isfile(os.path.join(n_path, f))]

    # read in NORMAL and PNEUMONIA
    images_pn = np.zeros((len(filelist_p), size[0]*size[1])) 
    images_n = np.zeros((len(filelist_n), size[0]*size[1])) 
    
    print("Reading PNEUMONIA images")
    for i in tqdm.tqdm(range(len(filelist_p))):
        im = Image.open(os.path.join(pn_path, filelist_p[i])).convert('L').resize(size)
        images_pn[i, :] = np.array(im).reshape((1, -1))

    print("Reading NORMAL images")
    for i in tqdm.tqdm(range(len(filelist_n))):
        im = Image.open(os.path.join(n_path, filelist_n[i])).convert('L').resize(size)
        images_n[i, :] = np.array(im).reshape((1, -1))

    # append labels
    images_n = np.concatenate((images_n, np.zeros((images_n.shape[0], 1))), axis=1)

    # pneumonia labels
    labels = np.array([1 if "bacteria" in f else 2 for f in filelist_p]).reshape((-1, 1))
    images_pn = np.concatenate((images_pn, labels), axis=1)

    total_data = np.concatenate((images_pn, images_n), axis=0)

    return total_data

if __name__ == "__main__":
    data = load_data()