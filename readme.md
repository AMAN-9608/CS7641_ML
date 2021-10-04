# CS 7641 Course Project

## Project Group 14: Amandeep Singh, Rajan V Tayal, Sai Shanbhag, Siddharth Sen, Yinghao Li

### Introduction
<p align="justify">
Pneumonia is an infection of one or both of the lungs in which the air sacs fill with pus and other liquid, and it is caused by bacteria, viruses, or fungi. Each year, pneumonia affects about 450 million people globally (7% of the population) and results in about 4 million deaths. Diagnosis is often based on symptoms and physical examination, and a Chest X-ray is one such screening technique, which may help confirm the diagnosis.</p>
<p align="center">
  <img width="300" height="196" src="https://miro.medium.com/max/1400/1*caVi5_pTsarvYlqkarijOg.png">
</p>
<p align = "center">
Illustrative Example of Chest X-Ray in Patients without Pneumonia (left) and with Pneumonia (right)
</p>
<p align="justify">
A qualified doctor then examines the X-ray for the signs of Pneumonia. Due to the subjectivity and manual bias of interpretations, the test results are not widely comparable. The current need is to establish a simple, automated, and objective screening technique which can adapt to a range of health and social service settings and would enable early detection of Pneumonia.</p>

### Problem Definition
<p align="justify">
The problem statement is to build a machine learning algorithm that accurately predicts whether the patient has Pnuemonia or not, based on their Chest X-ray.</p>

### Methods

### Potentials Results
<p align="justify">
The multi-label classification model shall have the following three label categories:
  
- Normal (No pneumonia)
- Bacterial Pneumonia
- Viral Pneumonia
<p align="justify">
Using the features extracted from the images, we shall use PCA to reduce dimensionality and visualize the data, and then use unsupervised learning such as clustering, to potentially identify 3 distinct clusters in our data (corresponding to our labels).
<p align="justify">
Next, we shall measure the performance of each of our supervised classification models using suitable metrics such as accuracy, precision, recall, AUC etc. to compare the models and conclusively identify the algorithm that works best for our classification task. 


### References
