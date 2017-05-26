## Visual Question Answering in Keras and Tensorflow Backend (by Fenil Doshi)

Implementation of the [VQA paper](https://arxiv.org/pdf/1505.00468v6.pdf). Website for Visual Question Answering -http://visualqa.org/

## Problem Description

Given an image and a natural Language question the task is to give a natural language answer. This is approached by encoding an image in a 4096 dimensional space which can be done by passing it through a VGG model. We will be removing the last 2 convolutional layers in order to get the required dimension. The question can be encoded in 2 ways:
- Bag of Words
- Using Recurrent Neural Networks
Once the question is encoded, the encoded image and encoded question are merged together and passes through a feed forward deep net. Finally we compute the answers from one of the 1000 classes(as we will take into account only the 1000 most frequently occuring answers).

![](https://github.com/feziodoshi/VQA/blob/master/keras%20implementation/data/vqa_image.png)

## Requirements
- Tensorflow
- Keras
- scipy
- spacy
- sklearn , numpy
- nltk
- NVIDIA CUDA 

download the spacy English glove vectors from https://nlp.stanford.edu/projects/glove/

## Dataset
Dataset Download link - http://visualqa.org/download.html
For more info on data preprocessing checkout the data folder in this directory

## References
- https://arxiv.org/pdf/1505.00468v6.pdf
- https://github.com/avisingh599/visual-qa
- https://github.com/anantzoid/VQA-Keras-Visual-Question-Answering
