---
layout: post
title: Landmark Retrieval
date: 2018-12-28 13:32:20 +0300
description: This is a technical report about my recent work in Chinese Academy of Sciences, focusing on Google Landmark Retrieval & Recognition task.
img: landmark.jpeg
tags: [Blog, DeepLearning]
author: David Ren
---

## **Attempts of Kaggle Competition: Landmark Retrieval**
<br/>

### **Chapter 1:&emsp;Prologue**
---
&emsp;This is a brief technical summary of what I have done and what I have learned these days in a battle with the kaggle challenge----Landmark Retrieval. Since the whole program is not finished yet and I’m just getting insight a little into the field of computer vision, I state here as an amateur just to clarify some difficulties I have encountered and accordingly tricky solutions. It is not an academic paper so I will focus myself on the chapter 3, 4, 5, not on the state-of-art methodologies used by other teams. The references are given in chapter 3 in an informal way and if there’s anything wrong, just contact me.
<br/>

### **Chapter 2: &emsp;Problem statement(Quotation of Kaggle)**
---
* **Description**

Image retrieval is a fundamental problem in computer vision: given a query image, can you find similar images in a large data base? This is especially important for query images containing landmarks, which accounts for a large portion of what people like to photograph.<br/>

In this competition, Kagglers are given query images and, for each query, are expected to retrieve all database images containing the same landmarks(if any).<br/>

* **Evaluation**

&emsp;Submissions are evaluated according to mean Average Precision @ 100(mAP @ 100):

mAP@100=1/Q ∑_(q=1)^Q▒1/(min⁡(m_q,100))  					∑_(k=1)^(min⁡(n_q,100))▒〖p_q (k)〖rel〗_q (k)〗

where:<br/>
* Q is the number of query images that depict landmarks from the index set<br/>
* mq is the number of index images containing a landmark in common with the query image q (note that this is only for queries which depict landmarks from the index set, so mq != 0)<br/>
* n_q is the number of predictions made by the system for query q<br/>
* p_q(k) is the precision at rank k for the q-th query<br/>
*〖rel〗_q(k) denotes the relevance of prediction k for the q-th query: it’s 1 if the k-th prediction is correct, and 0 otherwise.<br>
Some query images will have no associated index images to retrieve. These queries are ignored in scoring.<br/>

### **Chapter 3:&emsp;Methodologies and References**
---
* **Methodologies** <br/>
1.&emsp;Recognition Task1: [1]<br/>
2.&emsp;Recognition Task2: [2]<br/>
3.First price solution:&emsp;[http://www.kaggle.com/c/landmark-retrieval-challenge/discussion/57855](http://www.kaggle.com/c/landmark-retrieval-challenge/discussion/57855 "First Price solution")

* **References**<br/>
*[1]Large-Scale Image Retrieval with Attentive Deep Local Features<br/>
[2]Bilinear CNNs for Fine-grained Visual Recognition<br/>
[3]Deep Residual Learning for Image Recognition<br/>
[4]Very Deep Convolutional Networks for Large-scale Image Recognition<br/>
[5]Understanding the difficulty of training deep feedforward neural networks<br/>*

### **Chapter 4:&emsp;Method of mine**
---
(I skip intricate diagrams…)

Recent years have witnessed the flourish of deep CNNs in the field of computer vision. From AlexNet to VGG, Inception, ResNet, DenseNet. According to previous research, deeper convolutional neural network has better performance compared to simpler ones. Thus I decide to utilize ResNet50 as the base net in landmark retrieval challenge and the purpose is to test the performance of ResNet50 as feature extractor.

 ResNet [3], abbreviation of Residual CNN, is now a prevalent CNN with less parameters(nearly 1/6) compared to VGG16, so I believe it can achieve state-of-art performance like the paper [3] introduced and at the same time with as less time as possible.

As I will show you in the next chapters, using single ResNet to extract feature maps is disappointed, it is also intuitive to conduct that the classifier right after feature extraction is not robust enough. Instead, I originally decide to observe the classifying performance of a single CNN results a very poor accuracy, it suffers from severe overfitting: with training accuracy more than 90% and test accuracy less than 2%. The unreasonable low accuracy makes me to consider some improvements on the network structure. So here comes B-CNN (Bilinear Convolutional Neural Network).

For the attempts of B-CNN, I use a package of completely different scripts to examine the performance with ResNet50 as base net. Originally, the given base net is VGG16-D, which achieves test accuracy of 70% with fine-tuning the fc layer only, and 80% with fine-tuning all the layers. However, for ResNet50, the output of the last bottleneck is 2048d, which leads the dimension of the feature map after the model to be 2048*2048(512*512 for VGG). Unfortunately it suffers from memory overflow. Then I tried a smaller model----ResNet34. The results is not so charming but plausible enough to some extent, with training accuracy being up to 58% for fine-tuning fc only, finally it implicates that ResNet is not so suitable for bilinear model.

Followings are the attempts on the whole project:

* **Attempt 1. ResNet50 \-> feature map of 2048d \->classifier**
* **Bilinear module: ResNet50 -> feature map of 2048d ->  matrix dot multiplication
-> feature map of 2048\*2048 \-> full-connection layer -> feature map of 1000d -> classifier**<br/>

Here is the gathered data:

| Method| ResNet50-all | ResNet50-fc | Bilinear with ResNet50 | Bilinear with ResNet34 | Bilinear with VGG16 |
|-------------|--------------|------------|------------|-----------|----------|
|Training Acc (no DA)|>97%|>90%| |73.26%|88.86%|
|Test Acc (no DA)|<2%|<2%| |50.85%|57.49%|
|Training Acc (DA)| | | |68%|72.92%|
|Test Acc (DA)| | | |58%|70.59%|

<font size=1>Due to the time-consuming training process, I just use a single ResNet50 to examine the effect of classifying, and I fine-tune the fc layers and all the layers separately. The data set I used includes: 160000 training images of 1000 different landmarks and 39000 test images, both are derived from Google Landmark Retrieval Database. Notice that the Bilinear modules  are just an experiment of Bilinear CNN in order to examine the characteristics of it with ResNet as base net. The dataset used for this test is CUB_200 with 11788 images.</font>

### **Chapter 5:&emsp;Debugging Techniques**
---
* **About network selection (VGG or resnet)**

VGG [4], is designed for large scale image recognition with deep convolutional neural network structure, for different type it has different net configurations (A, B, C, D, E).  I choose VGG-D (Top-1 error 28.41, Top-5 error 9.62) as base net for this program.

ResNet [3], is designed for deeper feature learning, with currently state-of-art models exceeding 1000 layers. The residue mechanism makes the structure to learn more detailed features. Compared to VGG, ResNet has a surprising structure----- bottleneck, which used to reduce the amount of parameters during training process, thus correspondingly the time used for training ResNet is much less than other CNNs such as VGG and AlexNet with an even better performance. The structure I used is ResNet50 (Top-1 error 23.85, Top-5 error 7.13), with only 1/4  network parameters of VGG-D.

But when it comes to the difference of VGG-D and ResNet50 other than the training time, actually they have little difference in accuracy under ideal circumstance. However, for BCNN model I used in this program, it suggests that VGG is more suitable than ResNet to be the base net of BCNN. In fact pragmatically ResNet has overall better performance than VGG since it is deeper and faster. This puzzles me until now and providentially I find ResNet is not listed in the experiment in the paper of BCNN [2]. I have no way to know whether it is the poor ResNet in the BCNN model that makes it being cut off by the author.

* **About optimizer selection (Adam or SGD)**

In principle, Adam is a new algorithm claimed to supplant SGD, it adjust learning rate according to training process and thus leads to a faster convergence. In the experiment of VGG-based BCNN, it do lead to a faster performance significantly. However, I find in the experiments that Adam and SGD has nearly same performance in ResNet50 based BCNN, precisely speaking, they both leads to a pre-convergence near the training loss of 1.6~1.7, test accuracy of 58%, much poorer than VGG.

* **About input size (224\*224 or 448\*448)**

Before this, I list all input size configuration I used:<br/>
ResNet50 for LMR: 255\*255<br/>
ResNet34-based BCNN: 448\*448<br/>
ResNet34-based BCNN: 224\*224<br/>
VGG-D-based BCNN: 448\*448<br/>
VGG-D-based BCNN: 224\*224<br/>

Previously, I do not take input size into consideration, neither do PyTorch Docs. Normally we just use the size of 255, 224, 448. PyTorch Docs suggests to use “mini-batches of 3-channel RGB images of shape (3\*H\*W), where H and W are expected to be at least 224”. I have no way to know whether these models pre-trained on ImageNet require an input size of 224 or not. At the beginning, I choose the input size of 448 as the VGG-D-based BCNN and it works fine. Nevertheless when using ResNet50-based BCNN and ResNet34-based BCNN the results is so poor that only 35% accuracy is conducted.

![wuguan](https://i.imgur.com/Wo4e9Lr.jpg)

![](https://image-static.segmentfault.com/334/564/3345641267-59c9df4945ddc_articlex)

yes

















