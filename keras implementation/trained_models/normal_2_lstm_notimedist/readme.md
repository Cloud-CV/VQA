Description: 
- 2 stacked LSTM + CNN(4096 features) + Feed Forward(2 hidden layers)
- Hidden Units in LSTM model = 500
- Hidden Units in 1st hidden layer of Feed Forward Net = 1000
- Hidden Units in 2nd hidden layer of Feed Forward Net=500
- Activation function in 1st and 2nd hidden layers of Feed Forward Net= Relu
- Activation function in last layer= Softmax
- Loss function= Categorical Croos entropy
- Optimizer Used= Stochastic gradient descent
- Momentum=0.8
- No dropout and regularization

Model: [Link for json file](https://github.com/feziodoshi/VQA/blob/master/keras%20implementation/model/normal_2_lstm_nodistributed_2hidden.json)

Download Link: [Link for downloading the weights of this model](https://drive.google.com/open?id=0B_KG6xVZJiZtOHlUUFVNWXZHVHM)
