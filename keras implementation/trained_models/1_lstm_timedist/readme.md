Description: 
- 1 LSTM (with Time distributed Layer) + CNN(4096 features) + Feed Forward(2 hidden layers)
- Hidden Units in LSTM model = 500
- Hidden Units in 1st hidden layer of Feed Forward Net = 1000
- Hidden Units in 2nd hidden layer of Feed Forward Net=500
- Activation function in 1st and 2nd hidden layers of Feed Forward Net= Relu
- Activation function in last layer= Softmax
- Loss function= Categorical Croos entropy
- Optimizer Used= RMSProp
- No dropout and regularization

The LSTM outputs a 3d Tensor of Shape (None,23,50) which goes into a Time Distributed Dense Layer giving the shape (None,23,100). This is Reshaped into a 2d Layer of shape (None,23*100) and concatenated with the 4096 image dimensions. The rest of the architecture is the same as others. These type of networks work better with the RMSProp optimizer rather than SGD.

Model: [Link for json file](https://github.com/feziodoshi/VQA/blob/master/keras%20implementation/model/normal_2_lstm_nodistributed_2hidden.json)

Download Link: [Link for downloading the weights of this model](https://drive.google.com/open?id=0B_KG6xVZJiZtRUpoZnhUa3JhZUU)
