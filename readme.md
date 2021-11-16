# CS 7641 Course Project

> Project Group 14: Amandeep Singh, Rajan V Tayal, Sai Shanbhag, Siddharth Sen, Yinghao Li

## Proposal

### Introduction
<p align="justify">
Pneumonia is an infection of one or both lungs in which the air sacs fill with pus and other liquid, and it is caused by bacteria, viruses, or fungi. Each year, pneumonia affects 450 million people globally and results in about 4 million deaths. Diagnosis is often based on symptoms and physical examination, and a Chest X-ray is a screening technique, which may help confirm the diagnosis.</p>
<p align="center">
<img width="400" height="196" src="https://i.imgur.com/jZqpV51.png">
</p>
<p align = "center">
<font size="1">Illustrative Example of Chest X-Ray in Patients with No pneumonia, Bacterial Pneumonia and Viral Pneumonia</font>
</p>
<p align="justify">
A qualified doctor then examines the X-ray for the signs of Pneumonia. Due to the subjectivity and manual bias of interpretations, the test results are not widely comparable. The current need is to establish a simple, automated, and objective screening technique that can adapt to a range of health and social service settings and would enable early detection of Pneumonia.</p>

### Problem Definition
<p align="justify">
The problem statement is to build a machine learning algorithm that accurately predicts whether the patient has No pneumonia, Bacterial Pneumonia, or Viral Pneumonia, based on their Chest X-ray.</p>

### Methods

#### Supervised Learning

<p align="justify">
<!-- Supervised image classification is a fundamental and well-studied problem in the computer vision (CV) area. -->
<!-- The accuracy of recent models facilitates the wild-usage of image classification and image segmentation techniques in the real world. -->
<!-- In the pre-deep neural network (DNN) era, machine learning techniques such as decision tree or support vector machine (SVM). -->
With the development of computational power and mathematic algorithms, convolutional neural networks (CNN Cun et al. 1990) became the principal method to address supervised image classification.
Based on CNN, Deep CNN (DCNN Krizhevsky et al. 2012), Inception (Szegedy et al. 2015), deep residual network (ResNet He et al. 2016) or pre-trained models such as ImageBERT (Qi et al. 2020) keep pushing forward the frontier.</p>
<p align="justify">
We plan to implement a DCNN for this project. The guideline is to balance the model performance and complexity, including the time spent in implementing the model as well as the time for training the model and fine-tuning its hyper-parameters. The starting point is to follow Krizhevsky et al. 2020's work. We will consider adding residual connections to improve the model performance.</p>


#### Unsupervised Learning
<p align="justify">
Unsupervised learning is promising in the medical imaging field as it lacks the bias which is implicit in supervised learning (Raza, Khalid, and Nripendra 2021). Kernel Principal Component Analysis (PCA) has been used to successfully extract respiratory signal estimation from X-ray images (Fischer, Peter, et al. 2017). Another method is to apply Deep CNNs and use clustering algorithms on extracted features to group similar pictures (Cohn, Ryan, and Holm 2021). </p>
<p align="justify">
Our approach will involve reducing the dimensionality of the dataset, initially with PCA. Subsequent methods may attempt to utilize pre-trained image classifiers such as vgg16, AlexNet, or NiftyNet (Gibson, Eli, et al.). Finally, we will use a clustering algorithm such as k-means to group similar items together.</p>


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

### Proposed Timeline and Responsibilities
<p align="justify">
For the Midterm, we expect to be done with a baseline model that incorporates both supervised and unsupervised learning. Moving onwards to the Endterm, we'll spend our time comparing models and fine-tuning the selected model.<br>
For this proposal, all team members have contributed a similar amount of effort into researching related literature and drafting the different sections of the document.  Moving forward, we are going to distribute the work among the group – tentatively, 3 members (Amandeep, Siddharth, Yinghao) are going to work on the supervised learning aspect, while 2 members (Sai and Rajan) are going to primarily focus on the unsupervised learning aspect of the project. Task delegation is flexible and will be adjusted as needed; deliverables between the two groups will be communicated clearly to ensure the best possible result.</p>


## Midterm

### Methods

#### Unsupervised Approach: Principal Component Analysis

For the unsupervised method, the goal was to implement principal component analysis in order to achieve a better separation between the classes.
The first step was to resize, convert to grayscale, and normalize all the input images. The next step was to convert these images to <img src="https://render.githubusercontent.com/render/math?math=400 \times 400"> before applying PCA. Once that was done, we decided to retain 100 principal components which captured 86.59% of the variance.


#### Supervised Approach: Deep Convolutional Neural Network with Residual Connection

For the supervised method, we currently target the binary image classification task: has pneumonia or not.
To address this issue, we use a deep convolutional neural network (DCNN) with the residual connection that alleviates the gradient vanishing problem.
The model contains nine convolutional layers, which are put into three groups.

The first layer of each group doubles the number of the channels of the convolutional kernel, which is maintained by the following two layers.
A residual connection links the input and output of those two layers with addition operation.
One exception is the very first convolutional layer, which takes the one-channel input and produces a 32-channel output.
At the end of each group, we adopt a max polling layer with kernel size $2\times2$ and $2$ stride to reduce the image resolution by half.

The output of the convolutional groups is flattened and followed by three fully connected layers.
The output of the last fully connected layer is $2$-dimensional, which matches the number of classes.
In addition, we use a dropout ratio $0.1$ and ReLU activation function throughout the model.

All images are converted to $128\times128$ pixels with $1$ luminance channel before being fed into the model.
However, this number may change in later improvements.
A $5$-fold cross-validation is used to realize early stopping.

### Results

#### Supervised Approach
The model is trained with mini-batch gradient descent, Adam optimizer and linear learning rate scheduler with $0.2$ warmup ratio.
We use F1 score as the early stopping reference metric.
Within $100$ training epochs, the best F1 score the model achieves on the validation/training set is $0.9895$.
Other metrics of the best model are $0.9895$ precision, $0.9895$ recall and $0.9846$ accuracy.
On the test set, the metrics are: accuracy: $0.7692$; precision: $0.7312$; recall: $0.9974$;  f1: $0.8438$.
These results indicate severe over-fitting, which we will try to resolve in the following research.

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
