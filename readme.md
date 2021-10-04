# CS 7641 Course Project

## Project Group 14: Amandeep Singh, Rajan V Tayal, Sai Shanbhag, Siddharth Sen, Yinghao Li

### Introduction
<p align="justify">
Pneumonia is an infection of one or both of the lungs in which the air sacs fill with pus and other liquid, and it is caused by bacteria, viruses, or fungi. Each year, pneumonia affects about 450 million people globally (7% of the population) and results in about 4 million deaths. Diagnosis is often based on symptoms and physical examination, and a Chest X-ray is one such screening technique, which may help confirm the diagnosis.</p>
<p align="center">
  <img width="300" height="196" src="https://miro.medium.com/max/1400/1*caVi5_pTsarvYlqkarijOg.png">
</p>
<p align = "center">
<font size="1">Illustrative Example of Chest X-Ray in Patients without Pneumonia (left) and with Pneumonia (right)</font>
</p>
<p align="justify">
A qualified doctor then examines the X-ray for the signs of Pneumonia. Due to the subjectivity and manual bias of interpretations, the test results are not widely comparable. The current need is to establish a simple, automated, and objective screening technique which can adapt to a range of health and social service settings and would enable early detection of Pneumonia.</p>

### Problem Definition
<p align="justify">
The problem statement is to build a machine learning algorithm that accurately predicts whether the patient has Pnuemonia or not, based on their Chest X-ray.</p>

### Methods

#### Unsupervised Learning
<p align="justify">
Unsupervised learning is promising in the medical imaging field as it lacks the labelling and class creation bias which is implicit in supervised learning. Because unsupervised learning derives insights directly from data, it may be preferred for some applications (Raza, Khalid, and Nripendra 2021). Researchers have had success using Kernel Principal Component Analysis (PCA) to extract respiratory signal estimation from X-ray images (Fischer, Peter, et al., 2017).   In the wider image clustering field, state of the art algorithms apply Deep Convolutional Neural Networks and use standard clustering algorithms on extracted features to group similar pictures (Cohn, Ryan, and Holm, 2021). </p>
<p align="justify">
Our approach will involve reducing the dimensionality of the dataset (from an image of thousands of pixels to some tractable feature vector). We intend to apply PCA to scale down the size, but may attempt to utilize pre-trained image classifiers such as vgg16, AlexNet, or NiftyNet (Gibson, Eli, et al). Finally we will use a clustering algorithm such as k-means to group similar items together.</p>


### Potentials Results

The multi-label classification model shall have the following three label categories:

<ul>
<li>Normal (No pneumonia)</li>
<li>Bacterial Pneumonia</li>
<li>Viral Pneumonia</li>
</ul>

<p align="justify">
Using the features extracted from the images, we shall use PCA to reduce dimensionality and visualize the data, and then use unsupervised learning algorithms to potentially identify 3 distinct clusters in our data (corresponding to our labels).</p>
<p align="justify">
Next, we shall measure the performance of each of our supervised classification models using suitable metrics such as accuracy, precision, recall, AUC etc. to compare the models and conclusively identify the algorithm that works best for our classification task.</p>


### References
Raza, Khalid, and Nripendra Kumar Singh. “A Tour of Unsupervised Deep Learning for Medical Image Analysis.” [Link](https://doi.org/10.2174/1573405617666210127154257)<br>
Fischer, Peter, et al. “Unsupervised Learning for Robust Respiratory Signal Estimation from X-Ray Fluoroscopy.” IEEE Transactions on Medical Imaging, vol. 36, no. 4, 2017, pp. 865–877., https://doi.org/10.1109/tmi.2016.2609888. <br>
Cohn, Ryan, and Elizabeth Holm. “Unsupervised Machine Learning via Transfer Learning and K-Means Clustering to Classify Materials Image Data.” Integrating Materials and Manufacturing Innovation, vol. 10, no. 2, 2021, pp. 231–244., https://doi.org/10.1007/s40192-021-00205-8. <br>
Gibson, Eli, et al. “NiftyNet: A Deep-Learning Platform for Medical Imaging.” Computer Methods and Programs in Biomedicine, vol. 158, 2018, pp. 113–122., https://doi.org/10.1016/j.cmpb.2018.01.025. 
