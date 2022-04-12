# Handwritten-Chinese-Identifier
iOS app that uses a convolutional neural network to identify pictures of handwritten Chinese characters 

### Pretraining
First a base model using lenet-5 architecture was created using keras subclassing and pretrained on MNIST dataset. Lenet-5 involved ... 

### Transfer Learning & Fine Tuning
Then, transfer learning was used to fine tune the model on a custom image dataset collected of handwritten Chinese characters. Fine tuning involved freezing the convolutional layers, and retraining the last few fully connected layers on the custom dataset. This is because convolutional layers trained on the MNIST handwritten digits may also share some features with a custom handwritten dataset, while the last few FC layers are more specialized and need to be adapted to the new dataset through fine tuning. 

### Model Evaluation and Improvement 

### Convert to Tensorflow Lite Model

### 



