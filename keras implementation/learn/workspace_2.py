from keras.preprocessing.sequence import pad_sequences
import spacy
from keras.utils import np_utils
from keras import optimizers
import cv2
import numpy as np
import tensorflow as tf
# import scipy .io
from sklearn.externals import joblib
import vgg_19_workspace

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





def get_image_model(CNN_weights_file_name):
    ''' Takes the CNN weights file, and returns the VGG model update
    with the weights. Requires the file VGG.py inside models/CNN '''

    image_model = VGG_19(CNN_weights_file_name)


    sgd = SGD(lr=0.1, decay=0, momentum=0.8, nesterov=False)


    image_model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return image_model

def get_image_features(image_file_name, CNN_weights_file_name):
    ''' Runs the given image_file to VGG 16 model and returns the
    weights (filters) as a 1, 4096 dimension vector '''
    image_model = VGG_19(CNN_weights_file_name)

    image_features = np.zeros((1, 4096))
    # Magic_Number = 4096  > Comes from last layer of VGG Model

    # Since VGG was trained as a image of 224x224, every new image
    # is required to go through the same transformation
    im = cv2.resize(cv2.imread(image_file_name), (224, 224))
    im = im.transpose((2, 0, 1))  # convert the image to RGBA

    # this axis dimension is required because VGG was trained on a dimension
    # of 1, 3, 224, 224 (first axis is for the batch size
    # even though we are using only one image, we have to keep the dimensions consistent
    im = np.expand_dims(im, axis=0)

    image_features[0, :] = get_image_model(CNN_weights_file_name).predict(im)[0]
    return image_features

def get_image_features(image_id,im_map,vgg_pre_features):
    num_dim=4096
    num_samples=len(image_id)
    image_features=np.zeros((num_samples,num_dim))
    for i in range(len(image_id)):
        image_features[i,:]=vgg_pre_features[:,im_map[image_id[i]]]
    return image_features


# imgind_to_features=open('C:/Users/ezio/Desktop/vqa/features/coco_vgg_IDMap.txt','r').read().splitlines()
# vgg_model_path='C:/Users/ezio/Desktop/train2014/vgg_model/vgg_feats.mat'
# features_struct = scipy.io.loadmat(vgg_model_path)
# VGG_features=features_struct['feats']
# # x=get_image_features(img_file,weights)
# x=('524291', '524291')
# im_feat={}
# for i in imgind_to_features:
#     temp=i.split()
#     im_feat[temp[0]]=int(temp[1])
# l=get_image_features(x,im_feat,VGG_features)
# print(l.shape)
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