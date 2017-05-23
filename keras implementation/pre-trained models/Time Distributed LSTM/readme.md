## Architecture

Here the word vector of every word is collected in a 3-d Tensor-(None,<time sequence length-23>,<hidden dimensions of RNN=50>). This in turn is sent into a Dense Layer(Time Distributed Layer- thus again producing a 3d tensor but of higher dimension(about 100) of every word).This layer is then reshaped this layer into a normal 2d tensor of shape(None,23*100). Finally this is merged along with the image features(4096 features form VGG net)

Optimizer: RMSProp
Loss:Categorical Cross Entropy

## Weights

The weights of this model can be downloaded from https://drive.google.com/open?id=0B_KG6xVZJiZtRUpoZnhUa3JhZUU