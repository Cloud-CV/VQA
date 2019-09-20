from keras.models import Sequential,model_from_json
from keras.optimizers import SGD
import operator
from itertools import zip_longest
from collections import defaultdict
from sklearn.externals import joblib

import scipy.io
from learn.workspace_2 import get_answers_mat,recieve_batch,get_image_features
import spacy

word_embeddings=spacy.load('en',vectors='en_glove_cc_300_1m_vectors')
##helper functions
##this will just help me get the most frequently occuring questions and answers, we will also pass image train(the image id)
##we will ignore the question id for now
def selectFrequentAnswers(questions_train, answers_train, images_train, maxAnswers):
    answer_fq= defaultdict(int)
    #build a dictionary of answers
    for answer in answers_train:
        answer_fq[answer] += 1

    sorted_fq = sorted(answer_fq.items(), key=operator.itemgetter(1), reverse=True)[0:maxAnswers]
    top_answers, top_fq = zip(*sorted_fq)
    new_answers_train=[]
    new_questions_train=[]
    new_images_train=[]
    #only those answer which appear int he top 1K are used for training
    for answer,question,image in zip(answers_train, questions_train, images_train):
        if answer in top_answers:
            new_answers_train.append(answer)
            new_questions_train.append(question)
            new_images_train.append(image)

    return (new_questions_train,new_answers_train,new_images_train)

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)




##writing a function to get the embeddings of the question


##opening the required files
questions_id_train=open('data/text/questions_id_val2014.txt','r').read().splitlines()
images_train=open('data/text/images_val2014_all.txt','r').read().splitlines()
questions_train=open('data/text/questions_val2014.txt').read().splitlines()
answers_train=open('data/text/answers_val2014_all.txt','r').read().splitlines()
imgind_to_features=open('features/coco_vgg_IDMap.txt','r').read().splitlines()
vgg_model_path='C:/Users/ezio/Desktop/train2014/vgg_model/vgg_feats.mat'
##files to open to check training accuracy
# questions_id_train=open('data/text/questions_id_train2014.txt','r').read().splitlines()
# images_train=open('data/text/images_train2014.txt','r').read().splitlines()
# questions_train=open('data/text/questions_train2014.txt').read().splitlines()
# answers_train=open('data/text/answers_train2014_all.txt','r').read().splitlines()
# imgind_to_features=open('features/coco_vgg_IDMap.txt','r').read().splitlines()
# vgg_model_path='C:/Users/ezio/Desktop/train2014/vgg_model/vgg_feats.mat'


print("files loaded")
features_struct = scipy.io.loadmat(vgg_model_path)
VGG_features=features_struct['feats']

im_feat={}
for i in imgind_to_features:
    temp=i.split()
    im_feat[temp[0]]=int(temp[1])


label_encoder=joblib.load('model/label_encoder.pkl')
print('Done with embedder,ancoder,vgg feats')
no_classes=len(label_encoder.classes_)
img_dim=4096
quest_dim=500
total_epochs=100
SEQ_LENGTH=23
word_vec_dim=300
hidden_dim_feed=1000

model=model_from_json(open('model/1_lstm_timedistributed_2hidden.json').read())
# sgd=SGD(lr=0.01,momentum=0.8,decay=0.0,nesterov=False)
model.compile(optimizer='rmsprop',loss='categorical_crossentropy')
print("Your model is Loaded successfully")
model.load_weights('trained_models/1_lstm_timedist/_epoch_099.hdf5')
print('Weights are Loaded')
ans_pred=[]
# questions_train=questions_train[:200]
# answers_train=answers_train[:200]
# images_train=images_train[:200]
for qu_batch,an_batch,im_batch in zip(grouper(questions_train, 100, fillvalue=questions_train[-1]),
                                      grouper(answers_train, 100, fillvalue=answers_train[-1]),
                                      grouper(images_train, 100, fillvalue=images_train[-1])):
    # print(qu_batch)
    # print(im_batch)
    # print(an_batch)
    ##get all the matrices
    question_matrix=recieve_batch(qu_batch,word_embeddings)
    # answer_matrix=get_answers_mat(an_batch,label_encoder)
    image_matrix=get_image_features(im_batch,im_feat,VGG_features)
    prediction=model.predict_classes([image_matrix,question_matrix],verbose=0)
    ans_pred.extend(label_encoder.inverse_transform(prediction))

del word_embeddings
del VGG_features

print(type(ans_pred))
print(len(ans_pred))
print(ans_pred[0])
print(len(answers_train))
# input()
corrects=0.0
totals=0.0
for i in range(len(answers_train)):
    try:
        curr_answers=ans_pred[i].split(';')
        actual_answers=answers_train[i].split(';')
        curr_correct=0.0
        for j in curr_answers:
            if (j in actual_answers):
                # print("going")
                curr_correct += 1
        if (curr_correct > 4):
            corrects += 1
        totals += 1

    except:
        print("This is where you get an error")
        print(i)
        # input()


acc=corrects/totals
print("Acuracy is  ",acc)

#
#
# 	print(len(questions_train))
# '''
import gc; gc.collect()
