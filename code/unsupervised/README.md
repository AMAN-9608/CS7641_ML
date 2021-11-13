# PCA code

Included files:
- load_data.py: Takes data from the data folder and resizes each image to the inputed dimensions. Also extracts labels and linearizes each images. Outputs a Nx(window_size[0]*window_size[1] + 1) array
- pca.py: performs PCA on a dataset. Includes inputs for the window size and number of components. Optionally saves output to data/pca_transformed_data folder for eacy reloading.

It was easier to read the images into memory each time than keep them in memory for multiple PCA attempts. Reading images should take ~2 minutes while PCA can take up to 10. 

The output of pca_pipeline is a Nx(n_comp + 1) array, with the last column being the labels:
- 0 is NORMAL
- 1 is bacterial PNEUMONIA
- 2 is viral PNEUMONIA

# kMeans Code

Included files: kmeans.py
- takes data from PCA output and using the first 100 PCA components to perform k-means clustering. Accuracy score reported. 
- generates confusion matrix comparing predicted labels via clustering to the actual y labels
- generates visualization for the first 3 PCA components to see distribution of Pneumonia vs Non-Pneumonia images.
