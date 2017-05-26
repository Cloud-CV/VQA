from keras.layers import Dense,Activation,Dropout,Reshape,Merge
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM,GRU
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras.models import Sequential,model_from_json



##creating the model in Keras
img_dim=4096
quest_dim=500
total_epochs=100
SEQ_LENGTH=23
word_vec_dim=300
hidden_dim_feed=1000
no_classes=1000
#########################################################################

##will load the image model here
image_model=Sequential()
# image_model.add(Reshape(input_shape = (img_dim,), dims=(img_dim,)))
image_model.add(Reshape((img_dim,), input_shape=(img_dim,)))

##will import the language model here
questions_model=Sequential()
questions_model.add(GRU(output_dim=quest_dim,input_shape=(SEQ_LENGTH,word_vec_dim),return_sequences=True))
questions_model.add(GRU(output_dim=quest_dim,input_shape=(SEQ_LENGTH,word_vec_dim),return_sequences=False))


################################################################################

model=Sequential()
##merging image_dim and quest_dim
model.add(Merge([image_model,questions_model],mode='concat'))
model.add(Dense(output_dim=hidden_dim_feed,activation='relu',init='uniform'))
model.add(Dense(output_dim=500,activation='relu',init='uniform'))
model.add(Dense(output_dim=no_classes,activation='softmax'))
##because we are choosing 1000 most frequently occuring questions
# ##saving the model to json
name_model_mlp=model.to_json()
open('../model/normal_2_gru_nodistributed_2hidden.json','w').write(name_model_mlp)
print("Model is saved")

import gc ; gc.collect()