# Handwritten-Chinese-Identifier
iOS app that uses a convolutional neural network to identify pictures of handwritten Chinese characters 

### Pretraining
A base model using Lenet-5 architecture was created with keras subclassing and pretrained on MNIST dataset. The architecture consists of inputs 28x28 fed into a 5x5x6 conv layer --> average pooling layer --> 5x5x16 conv layer --> average pooling layer --> flatten --> fc 120 --> fc 84 --> softmax 10 classes

[add architecture graph]

### Transfer Learning & Fine Tuning
Then, transfer learning was used to fine tune the model on a custom image dataset collected of handwritten Chinese characters. Fine tuning involved freezing the convolutional layers, and retraining the last few fully connected layers on the custom dataset. This is because convolutional layers trained on the MNIST handwritten digits may also share some features with a custom handwritten dataset, while the last few FC layers are more specialized and need to be adapted to the new dataset through fine tuning. 

### Model Evaluation and Improvement 

[add loss plots] 

### Convert TF model to CoreML model 



