from numpy.core.numeric import full
from sklearn.decomposition import PCA
import numpy as np
from load_data import load_data
import os

save_dir = os.path.join("..", "..", "data", "pca_transformed_data")

def pca_transformation(data,n_comp):
    data_no_labels = data[:,:-1]
    pca = PCA(n_comp)
    transformed_data = pca.fit_transform(data_no_labels)
    print("Preserved variance = {}".format(np.sum(pca.explained_variance_ratio_)))
    return np.hstack((transformed_data,data[:,-1].reshape((len(data),1))))

def pca_pipeline(resize=(400,400), n_comp=100, save=True):
    data = load_data(resize)
    print("PCA with n_comp = {}".format(n_comp))
    pca_trans = pca_transformation(data, n_comp)
    if save:
        filename = "pca_data_{}_{}_{}.npy".format(resize[0], resize[1], n_comp)
        full_save_path = os.path.join(save_dir, filename)
        with open(full_save_path, 'wb') as f:
            np.save(f, pca_trans)
        print("Data saved to: {}".format(full_save_path))
    return pca_trans

if __name__ == "__main__":
    pca_pipeline(resize=(300,300))