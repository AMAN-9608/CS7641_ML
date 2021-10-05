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

#### Supervised Learning

<p align="justify">
<!-- Supervised image classification is a fundamental and well-studied problem in the computer vision (CV) area. -->
<!-- The accuracy of recent models facilitates the wild-usage of image classification and image segmentation techniques in the real world. -->
<!-- In the pre-deep neural network (DNN) era, machine learning techniques such as decision tree or support vector machine (SVM). -->
With development of computational power and mathematic algorithms, convolutional neural networks (CNN <a href="#XCun.1990.Handwritten">Cun et&#x00A0;al.</a>,&#x00A0;<a href="#XCun.1990.Handwritten">1990</a>) became the principle method to address supervised image classification.
Based on CNN, Deep CNN (DCNN <a href="#XKrizhevsky.2012.ImageNet">Krizhevsky et&#x00A0;al.</a>,&#x00A0;<a href="#XKrizhevsky.2012.ImageNet">2012</a>), Inception [<a href="#XSzegedy.2015.inception">Szegedy et&#x00A0;al.</a>,&#x00A0;<a href="#XSzegedy.2015.inception">2015</a>], deep residual network (ResNet <a href="#XHe.2016.resnet">He et&#x00A0;al.</a>,&#x00A0;<a href="#XHe.2016.resnet">2016</a>) or pre-trained models such as ImageBERT [<a href="#Xqi.2020.imagebert">Qi et&#x00A0;al.</a>,&#x00A0;<a href="#Xqi.2020.imagebert">2020</a>] keep pushing forward the frontier.</p>
<p align="justify">
We plan to implement a DCNN for this project.
The guideline is to balance the model performance and complexity, including the time spent in implementing the model as well as the time for training the model and fine-tuning its hyper-parameters.
The starting point is to follow <a href="#XKrizhevsky.2012.ImageNet">Krizhevsky et&#x00A0;al.</a>&#x00A0;[<a href="#XKrizhevsky.2012.ImageNet">2012</a>]&#8217;s work, by we will consider to add residual connections to improve the model performance.</p>


#### Unsupervised Learning
<p align="justify">
Unsupervised learning is promising in the medical imaging field as it lacks the labelling and class creation bias which is implicit in supervised learning. Because unsupervised learning derives insights directly from data, it may be preferred for some applications (Raza, Khalid, and Nripendra 2021). Researchers have had success using Kernel Principal Component Analysis (PCA) to extract respiratory signal estimation from X-ray images (Fischer, Peter, et al., 2017). In the wider image clustering field, state of the art algorithms apply Deep Convolutional Neural Networks and use standard clustering algorithms on extracted features to group similar pictures (Cohn, Ryan, and Holm, 2021). </p>
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
Fischer, Peter, et al. “Unsupervised Learning for Robust Respiratory Signal Estimation from X-Ray Fluoroscopy.” [Link](https://doi.org/10.1109/tmi.2016.2609888) <br>
Cohn, Ryan, and Elizabeth Holm. “Unsupervised Machine Learning via Transfer Learning and K-Means Clustering to Classify Materials Image Data.” [Link](https://doi.org/10.1007/s40192-021-00205-8) <br>
Gibson, Eli, et al. “NiftyNet: A Deep-Learning Platform for Medical Imaging.” [Link](https://doi.org/10.1016/j.cmpb.2018.01.025)

<div class="thebibliography">
<p class="bibitem" ><span class="biblabel">
<a id="XCun.1990.Handwritten"></a><span class="bibsp">&#x00A0;&#x00A0;&#x00A0;</span></span>Y.&#x00A0;Le Cun, B.&#x00A0;Boser, J.&#x00A0;S. Denker, R.&#x00A0;E. Howard, W.&#x00A0;Habbard, L.&#x00A0;D. Jackel, and D.&#x00A0;Henderson. <span class="ptmri8t-">Handwritten</span>
<span class="ptmri8t-">Digit Recognition with a Back-Propagation Network</span>, pages 396&#8211;404. Morgan Kaufmann Publishers Inc., San
Francisco, CA, USA, 1990. ISBN 1558601007.
</p>
<p class="bibitem" ><span class="biblabel">
<a id="XKrizhevsky.2012.ImageNet"></a><span class="bibsp">&#x00A0;&#x00A0;&#x00A0;</span></span>Alex Krizhevsky, Ilya Sutskever, and Geoffrey&#x00A0;E Hinton. Imagenet classification with deep convolutional neural networks. In F.&#x00A0;Pereira, C.&#x00A0;J.&#x00A0;C. Burges, L.&#x00A0;Bottou, and K.&#x00A0;Q. Weinberger, editors,
<span class="ptmri8t-">Advances in Neural Information Processing Systems</span>, volume&#x00A0;25. Curran Associates, Inc., 2012. <a href="https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf" class="url" ><span class="ectt-1000">URL</span></a>.
</p>
<p class="bibitem" ><span class="biblabel">
<a id="XSzegedy.2015.inception"></a><span class="bibsp">&#x00A0;&#x00A0;&#x00A0;</span></span>Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan,
Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In <span class="ptmri8t-">2015 IEEE Conference on</span>
<span class="ptmri8t-">Computer Vision and Pattern Recognition (CVPR)</span>, pages 1&#8211;9, 2015. doi:<a href="https://doi.org/10.1109/CVPR.2015.7298594" >10.1109/CVPR.2015.7298594</a>.
</p>
<p class="bibitem" ><span class="biblabel">
<a id="XHe.2016.resnet"></a><span class="bibsp">&#x00A0;&#x00A0;&#x00A0;</span></span>Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition.
In <span class="ptmri8t-">2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)</span>, pages 770&#8211;778, 2016.
doi:<a href="https://doi.org/10.1109/CVPR.2016.90" >10.1109/CVPR.2016.90</a>.
</p>
<p class="bibitem" ><span class="biblabel">
<a id="Xqi.2020.imagebert"></a><span class="bibsp">&#x00A0;&#x00A0;&#x00A0;</span></span>Di&#x00A0;Qi, Lin Su, Jia Song, Edward Cui, Taroon Bharti, and Arun Sacheti. Imagebert: Cross-modal pre-training with
large-scale weak-supervised image-text data, 2020.
</p>
</div>
