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

Included file: kmeans.py
- takes data from PCA output and using the first 100 PCA components to perform k-means clustering. Accuracy score reported. 
- generates confusion matrix comparing predicted labels via clustering to the actual y labels
  ![image](https://user-images.githubusercontent.com/40197136/141605798-074e9fad-5375-40a8-90b7-99984732d39a.png)
- generates visualization for the first 3 PCA components to see distribution of Pneumonia vs Non-Pneumonia images.
  ![image](https://user-images.githubusercontent.com/40197136/141605809-b95cebe4-7e5a-469a-804d-343fd840bea1.png)
  
  
# Supervised Classification Results 
  
 We used the first 100 principal components to build supervised classification models in order to classify the images as ‘normal’ or ‘viral’. Reducing the number of features in the flattened image (400 x 400) from 1,60,000 to 100 with PCA will help reduce overfitting on our training dataset of 4172 images.

Support Vector Classifier, Random Forest Classifier and Logistic Regression were the models which we tested on our training dataset which had dimensions of 4172 x 100. A grid search based 5-fold cross validation routine was used to fine-tune the hyperparameters of these models. Model performance metrics on test dataset are shown in the table below :


| Model    | Precision | Recall | Accuracy Score |
-----------|-----------|--------|----------------|
  SVM      | 0.9690    | 0.9728 |    0.9568      
  RF       | 0.9845    | 0.9305 |    0.9339      
  Logistic | 0.9690    | 0.9728 |    0.9568      

 
From the table above, we can see that Logistic Regression and Support Vector classifier outperform Random forest classifier in recall and accuracy score metrics. As a next step, we will re-train these models on features obtained via t-sne; another dimensionality reduction technique to select the best feature reduction method for our dataset. We also plan to develop a web based interactive framework to deploy our trained models and classify new incoming images as normal or viral.

