# Handwritten-Chinese-Identifier
iOS app that uses a convolutional neural network to identify pictures of handwritten Chinese characters 


### Neural Network Architecture 

https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=726791 

An implementation of Lenet-5, the classic convolutional network used to identify handwritten numerical digits, will be trained to identify handwritten chinese characters. The model's architecture is as follows: 

input = 32 x 32 pixel image 

- c1 = convlayer f = 5x5x6, s = 1, activation = tanh
- S2 = average pooling 
- c3 = convlayer f = 5x5x16, s = 1, activation = tanh
- s4 = average pooling 
- flatten 
- FC5 = dense 120, activation = tanh
- FC6 = dense 84, activation = tanh
- FC7 = dense 10, activation = softmax 

