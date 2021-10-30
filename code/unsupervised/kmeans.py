# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 17:51:26 2021

@author: siddh
"""

import numpy as np
import sklearn
import os
import matplotlib.pyplot as plt

pca_data = np.load(r"..\..\data\pca_transformed_data\pca_data_300_300_100.npy")

pca_data.shape
print(np.unique(pca_data[:, -1]))

#%%

from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

X = pca_data[:, 0:3]
y = pca_data[:, -1]
y = np.array([1. if d == 2 else d for d in y])

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)
y_pred = kmeans.labels_

homogeneity_score(y, y_pred)

# permutations = [[0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]]
permutations = [[0,1], [1,0]]

for perm in permutations:
    y_perm = np.choose(y_pred, perm)
    print("For", perm, "the accuracy is:", sum([y[i] == y_perm[i] for i in range(len(y))])/len(y))

# y_perm = np.choose(y_pred, permutations[2])

#%%

# Compute confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()

y_matched = np.choose(y_pred, [0,1])
cm = confusion_matrix(y, y_matched)

sns.heatmap(cm, square=True, annot=True, fmt='d', cbar=True, cmap="YlGnBu", xticklabels=[0,1], yticklabels=[0,1])

#%%

import plotly.graph_objects as go

# Generate the values
x_vals = pca_data[:,0]
y_vals = pca_data[:,1]
z_vals = pca_data[:,2]


fig = go.Figure(data=[go.Scatter3d(x=x_vals, y=y_vals, z=z_vals,mode='markers',
    marker=dict(
        size=3,
        color=y,                # set color to an array/list of desired values
        colorscale='Bluered',   # choose a colorscale
        opacity=0.8
    ))])

fig.update_layout(
    autosize=False,
    width=1000,
    height=1000,
    paper_bgcolor="LightSteelBlue",
)

fig.show(showlegend=True)