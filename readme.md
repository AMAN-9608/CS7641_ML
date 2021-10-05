# CS 7641 Course Project

## Project Group 14: Amandeep Singh, Rajan V Tayal, Sai Shanbhag, Siddharth Sen, Yinghao Li

### Introduction
<p align="justify">
Pneumonia is an infection of one or both of the lungs in which the air sacs fill with pus and other liquid, and it is caused by bacteria, viruses, or fungi. Each year, pneumonia affects 450 million people globally and results in about 4 million deaths. Diagnosis is often based on symptoms and physical examination, and a Chest X-ray is a screening technique, which may help confirm the diagnosis.</p>
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
The problem statement is to build a machine learning algorithm that accurately predicts whether the patient has No pneumonia, Bacterial Pneumonia or Viral Pneumonia, based on their Chest X-ray.</p>

### Methods

#### Supervised Learning

<p align="justify">
<!-- Supervised image classification is a fundamental and well-studied problem in the computer vision (CV) area. -->
<!-- The accuracy of recent models facilitates the wild-usage of image classification and image segmentation techniques in the real world. -->
<!-- In the pre-deep neural network (DNN) era, machine learning techniques such as decision tree or support vector machine (SVM). -->
With development of computational power and mathematic algorithms, convolutional neural networks (CNN Cun et al. 1990) became the principle method to address supervised image classification.
Based on CNN, Deep CNN (DCNN Krizhevsky et al. 2012), Inception (Szegedy et al. 2015), deep residual network (ResNet He et al. 2016) or pre-trained models such as ImageBERT (Qi et al. 2020) keep pushing forward the frontier.</p>
<p align="justify">
We plan to implement a DCNN for this project.
The guideline is to balance the model performance and complexity, including the time spent in implementing the model as well as the time for training the model and fine-tuning its hyper-parameters.
The starting point is to follow Krizhevsky et al. 2020's work, by we will consider to add residual connections to improve the model performance.</p>


#### Unsupervised Learning
<p align="justify">
Unsupervised learning is promising in the medical imaging field as it lacks the bias which is implicit in supervised learning (Raza, Khalid, and Nripendra 2021). Researchers have had success using Kernel Principal Component Analysis (PCA) to extract respiratory signal estimation from X-ray images (Fischer, Peter, et al. 2017). In the wider image clustering field, state of the art algorithms apply D CNNs and use clustering algorithms on extracted features to group similar pictures (Cohn, Ryan, and Holm 2021). </p>
<p align="justify">
Our approach will involve reducing the dimensionality of the dataset, initially with PCA. Subsequent methods may attempt to utilize pre-trained image classifiers such as vgg16, AlexNet, or NiftyNet (Gibson, Eli, et al). Finally, we will use a clustering algorithm such as k-means to group similar items together.</p>


### Potentials Results

The multi-label classification model shall have the following three label categories:

<ul>
<li>Normal (No pneumonia)</li>
<li>Bacterial Pneumonia</li>
<li>Viral Pneumonia</li>
</ul>

<p align="justify">
Using the features extracted from the images, we shall use PCA to reduce dimensionality and visualize the data, and then use unsupervised learning to potentially identify 3 distinct clusters in our data (corresponding to our labels).</p>
<p align="justify">
Next, we shall measure the performance of each of our supervised classification models using suitable metrics such as accuracy, precision, recall, AUC etc. to compare the models and conclusively identify the algorithm that works best for our classification task.</p>

### Proposed Timeline

### References
Raza, Khalid, and Nripendra Kumar Singh. [A Tour of Unsupervised Deep Learning for Medical Image Analysis](https://doi.org/10.2174/1573405617666210127154257)<br>
Fischer, Peter, et al. [Unsupervised Learning for Robust Respiratory Signal Estimation from X-Ray Fluoroscopy](https://doi.org/10.1109/tmi.2016.2609888) <br>
Cohn, Ryan, and Elizabeth Holm. [Unsupervised Machine Learning via Transfer Learning and K-Means Clustering to Classify Materials Image Data](https://doi.org/10.1007/s40192-021-00205-8) <br>
Gibson, Eli, et al. [NiftyNet: A Deep-Learning Platform for Medical Imaging](https://doi.org/10.1016/j.cmpb.2018.01.025)<br>
Y. Le Cun, B. Boser, J. S. Denker, R. E. Howard, W. Habbard, L. D. Jackel, and D. Henderson. [Handwritten Digit Recognition with a Back-Propagation Network](https://dl.acm.org/doi/10.5555/2969830.2969879)<br>
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. [Imagenet classification with deep convolutional neural networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) <br>
Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. [Going deeper with convolutions](https://doi.org/10.1109/CVPR.2015.7298594) <br>
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. [Deep residual learning for image recognition](https://doi.org/10.1109/CVPR.2016.90) <br>
Di Qi, Lin Su, Jia Song, Edward Cui, Taroon Bharti, and Arun Sacheti. [Imagebert: Cross-modal pre-training with large-scale weak-supervised image-text data](https://arxiv.org/abs/2001.07966)
