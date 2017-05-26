import matplotlib.pyplot as plt
from keras.layers import Dense,Activation,Dropout,Reshape,Merge
from keras.layers.recurrent import LSTM,GRU
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras.models import Sequential,model_from_json
import operator
from itertools import zip_longest
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
from keras import backend as K
import scipy.io
import numpy as np
from nltk.tokenize import word_tokenize
from learn.workspace_2 import get_answers_mat,recieve_batch,get_image_features
from random import shuffle
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
questions_id_train=open('data/text/questions_id_train2014.txt','r').read().splitlines()
images_train=open('data/text/images_train2014.txt','r').read().splitlines()
questions_train=open('data/text/questions_train2014.txt').read().splitlines()
answers_train=open('data/text/answers_train2014_all.txt','r').read().splitlines()
imgind_to_features=open('features/coco_vgg_IDMap.txt','r').read().splitlines()
vgg_model_path='C:/Users/ezio/Desktop/train2014/vgg_model/vgg_feats.mat'
# vgg_model_path='C:/Users/ezio/Desktop/train2014/vgg_model/vgg_feats.mat'


maxAnswers=1000
questions_train, answers_train, images_train = selectFrequentAnswers(questions_train,answers_train,images_train, maxAnswers)
# len_questions=[]
# for i in questions_train:
# 	len_questions.append(len(word_tokenize(i)))
# print(max(len_questions))
# '''
features_struct = scipy.io.loadmat(vgg_model_path)
VGG_features=features_struct['feats']

im_feat={}
for i in imgind_to_features:
	temp=i.split()
	im_feat[temp[0]]=int(temp[1])
# '''
#################checking#####################
# print(type(answers_train))
# print(len(answers_train))
# print(answers_train[0])
##############################################

#####################encode if not done######################
##encoding the answers
# label_encoder=LabelEncoder()
# label_encoder.fit(answers_train)
# print(len(set(answers_train))==len(label_encoder.classes_))

##saving the classifier
# joblib.dump(label_encoder,'model/label_encoder.pkl')
##############################################################


label_encoder=joblib.load('model/label_encoder.pkl')
print('Done with embedder,ancoder,vgg feats')
no_classes=len(label_encoder.classes_)
# print(no_classes)

# question_matrix = recieve_batch(questions_train, word_embeddings)
# del word_embeddings
# features_struct = scipy.io.loadmat(vgg_model_path)
# VGG_features=features_struct['feats']


# bad way of managing memory
# im_feat={}
# for i in imgind_to_features:
# 	temp=i.split()
# 	im_feat[temp[0]]=int(temp[1])
#
# answer_matrix=get_answers_mat(answers_train,label_encoder)
# image_matrix=get_image_features(images_train,im_feat,VGG_features)
#
# del VGG_features
# print("Loaded entire data")
# input()

#############################################################
# ##creating the model in Keras
# img_dim=4096
# quest_dim=500
# total_epochs=100
# SEQ_LENGTH=23
# word_vec_dim=300
# # hidden_dim_feed=2000

model=model_from_json(open('model/1_lstm_timedistributed_2hidden.json').read())
# sgd=SGD(lr=0.01,momentum=0.8,decay=0.0,nesterov=False)
model.compile(optimizer='rmsprop',loss='categorical_crossentropy')
# plot(model,to_file='mlp_first.png')
# model is loaded
print("Your model is Loaded successfully")
# model.load_weights('trained_models/normal_2_lstm_timedist/_epoch_012_loss_1.664.hdf5')
# print('Weights are Loaded')
# call_chkpt=ModelCheckpoint('trained_models/chkpoint/weights.{epoch:02d}-{loss:.2f}.hdf5',monitor='loss',save_best_only=True,mode='min')
# call_tensorboard=TensorBoard(log_dir='trained_models', histogram_freq=0)
# let us start the training procedure
# writer=tf.summary.FileWriter("/tmp/mnist_demo/1")
num_examples=10000
loss_arr=[]
loss=0.0
for epoch in range(100):

	for qu_batch,an_batch,im_batch in zip(grouper(questions_train[:num_examples], 20, fillvalue=questions_train[-1]),
										  grouper(answers_train[:num_examples], 20, fillvalue=answers_train[-1]),
										  grouper(images_train[:num_examples], 20, fillvalue=images_train[-1])):
		##get all the matrices
		question_matrix=recieve_batch(qu_batch,word_embeddings)
		answer_matrix=get_answers_mat(an_batch,label_encoder)
		image_matrix=get_image_features(im_batch,im_feat,VGG_features)
		# print("Done_111111!!!!!!!!!!!!")
		# model.fit([image_matrix,question_matrix],answer_matrix,batch_size=10,nb_epoch=1,callbacks=[call_chkpt,call_tensorboard])

		loss=model.train_on_batch([image_matrix,question_matrix],answer_matrix)

	# input()
	# 		##now get all the features
	loss_arr.append(loss)
	model.save_weights('trained_models/1_lstm_timedist/_epoch_{:03d}_loss_{:.4f}.hdf5'.format(epoch,loss),overwrite=True)
	print(epoch,">> ","loss",loss)
	joblib.dump(loss_arr,'trained_models/1_lstm_timedist/loss_arr.pkl')
#
#
# 	print(len(questions_train))
# '''
import gc; gc.collect()
