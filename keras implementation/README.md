## Visual Question Answering in Keras and Tensorflow Backend (by Fenil Doshi)

Implementation of the [VQA paper](https://arxiv.org/pdf/1505.00468v6.pdf). Website for Visual Question Answering -http://visualqa.org/

## Problem Description

Given an image and a natural Language question the task is to give a natural language answer. This is approached by encoding an image in a 4096 dimensional space which can be done by passing it through a VGG model. We will be removing the last 2 max pool layers in order to get the required dimension. The question can be encoded in 2 ways:
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

## To get started
Instruction in readme files in every folder

## Results
The 2 stacked GRU+CNN model converged faster than the corresponding LSTM over the same training set. I also figured out that the SGD optimizer worked better than RMSProp in  normal LSTMs/GRUs but RMSProp worked better in case of a Time Distributed Layer. GRU with the time distributed layer gave a very low accuracy. 

The models can be improvised way further by training it on the entire dataset for about >100 epochs on a better GPU(Tesla or GTX 1080 ). Overfitting can further be reduced by using Dropout and Regularization. 


1. Validation Accuracy of LSTM + CNN = 33.77 %
2. Validation Accuracy of GRU + CNN = 34.4 %
3. Validation Accuracy of LSTM + Time Distributed Layer + CNN = 34.3 %

These accuracies are by training over just 10,000 examples. The accuracy can be improved by training over a larger set every epoch and over a better GPU. Currently trained it on a NVIDIA GTX 960M which took around 3 hours to train 10,000 images for 100 epochs.

## Some Improvements that can be made
- Better hyperparameter tuning
- Dropout
- Regularization
- Using a RNN decoder for answers to get answers with temporal semantics

## References
- https://arxiv.org/pdf/1505.00468v6.pdf
- https://github.com/avisingh599/visual-qa
- https://github.com/anantzoid/VQA-Keras-Visual-Question-Answering
