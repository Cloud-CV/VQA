from keras.preprocessing.sequence import pad_sequences
import spacy
from keras.utils import np_utils
from keras import optimizers
import cv2
import numpy as np
import tensorflow as tf
# import scipy .io
from sklearn.externals import joblib
from learn.vgg_19_workspace import VGG_19

tf.python.control_flow_ops = tf
print("Started")

def get_answers_mat(answers_batch,labelencoder):
    y=labelencoder.transform(answers_batch)
    no_classes=labelencoder.classes_.shape[0]
    y=np_utils.to_categorical(y,no_classes)
    return y


def get_curr_question_tensor(curr_question,word_embeddings):
    tokens=word_embeddings(curr_question)
    curr_question_tensor=np.zeros((300,))
    for i in tokens:
        curr_question_tensor=np.vstack((curr_question_tensor,i.vector))
    return curr_question_tensor[1:,:]


def recieve_batch(batch_questions,word_embeddings):
    ##take a question and pass it to get_curr_question tensor
    batch_question_tensor=[]
    for i in batch_questions:
        batch_question_tensor.append(get_curr_question_tensor(i,word_embeddings))
    ##this part is wrong and will be removed later
    # batch_question_tensor=np.array(batch_question_tensor)
    # print(batch_question_tensor.shape)
    batch_question_tensor=pad_sequences(batch_question_tensor, maxlen=23)
    # print(batch_question_tensor.shape)
    return(batch_question_tensor)

# x=get_curr_question_tensor('Hi my name is Fenil')
# print(x.shape)
# recieve_batch(('Hi my name is Fenil','Hi my name is'))






def get_image_features(image_id,im_map,vgg_pre_features):
    num_dim=4096
    num_samples=len(image_id)
    image_features=np.zeros((num_samples,num_dim))
    for i in range(len(image_id)):
        image_features[i,:]=vgg_pre_features[:,im_map[image_id[i]]]
    return image_features


'''


label_encoder=joblib.load('C:/Users/ezio/Desktop/vqa/model/label_encoder.pkl')
print('Done')
no_classes=len(label_encoder.classes_)



answers=('orange;orange;orange;orange;orange;orange;orange;orange;orange;orange', 'yes;yes;yes;yes;yes;no;yes;yes;yes;yes')
y=get_answers_mat(answers_batch=answers,labelencoder=label_encoder)
for i in y[1]:
    if(i==1):
        print("Yes")
import gc; gc.collect()
'''
