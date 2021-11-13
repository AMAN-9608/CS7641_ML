import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

read_file = os.path.join("..", "..", "data", "pca_transformed_data", "pca_data_300_300_100.npy")

def readPCA(file):
    return np.load(file)

def makeBinaryLabels(y):
    return np.array([1. if d == 2 else d for d in y])
    
def visualizePCA(file, save=True):
    pca_data = readPCA(file)
    
    x_vals = pca_data[:,0]
    y_vals = pca_data[:,1]
    z_vals = pca_data[:,2]
    
    y = makeBinaryLabels(pca_data[:, -1])
    
    fig = plt.figure(figsize=(16,9), dpi=100)
    ax = fig.add_subplot(projection='3d')
    
    ax.scatter(x_vals[y==1], y_vals[y==1], z_vals[y==1], c='#3182bd', label='1', s=3)
    ax.scatter(x_vals[y==0], y_vals[y==0], z_vals[y==0], c='#e6550d', label='0', s=3)
    
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    
    fig.savefig('pca_visualization.png')
    print("Visualized PCA")

def kMeans(file, save=True):
    pca_data = readPCA(file)
    
    X = pca_data[:, 0:100]
    y = makeBinaryLabels(pca_data[:, -1])
    
    # Fit kMeans
    print("Fitting kMeans clustering...")
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)
    y_pred = kmeans.labels_
    
    # Compute Accuracy
    permutations = [[0,1], [1,0]]
    acc = []
    for perm in permutations:
        y_perm = np.choose(y_pred, perm)
        acc.append(sum([y[i] == y_perm[i] for i in range(len(y))])/len(y))
    print("Accuracy of kMeans = {:.3f}".format(max(acc)))
    
    # Generate Confusion Matrix
    perm = permutations[acc.index(max(acc))]
    y_matched = np.choose(y_pred, [0,1])
    cm = confusion_matrix(y, y_matched)
    
    cm_plot = sns.heatmap(cm, square=True, annot=True, fmt='d', cbar=True, cmap="YlGnBu", xticklabels=[0,1], yticklabels=[0,1])  
    if save:
        fig = cm_plot.get_figure()
        fig.savefig("kmeans_confusionmatrix.png")
    
        
if __name__ == "__main__":
    kMeans(read_file)
    visualizePCA(read_file)

