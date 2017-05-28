Contains 3 scripts:

- free_mem.py : Free GPU memory as and when needed
- workspace_2.py : Script containing functions to get the image computed features, question tensor (to feed it in the LSTM) , and also load the features directly from the vgg 19 model(**[IMPORTANT]**: Since training is done on VGG 16, try not to use this code as it will give error if incorrect weights are loaded or low accuracy even if correct weights are loaded)

- vgg_19_workspace.py: Constructs VGG 19 layer and removes the top two layers as we are training on 4096 image features. This code is inspired from https://github.com/thtrieu/essence/blob/master/src/utils/VGG.py which os the same for VGG 16
