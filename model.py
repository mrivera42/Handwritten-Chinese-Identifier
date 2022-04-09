import tensorflow as tf
import numpy as np


class LeNet5(tf.keras.Model):

    def __init__(self):
        super().__init__()

        self.c1 = tf.keras.layers.Conv2D(
            filters=6,
            kernel_size=(5,5),
            strides=(1,1),
            activation='tanh')
        self.s2 = tf.keras.layers.AveragePooling2D()
        self.c3 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(5,5),
            strides=(1,1),
            activation='tanh')
        self.s4 = tf.keras.layers.AveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.fc5 = tf.keras.layers.Dense(120,activation='tanh')
        self.fc6 = tf.keras.layers.Dense(120,activation='tanh')

    def call(self,inputs):

        x = self.c1(inputs)
        x = self.s2(x)
        x = self.c3(x)
        x = self.s4(x)
        x = self.flatten(x)
        x = self.fc5(x)
        x = self.fc6(x)

if __name__ == '__main__':
    
    # load dataset 
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    # preprocess 
    x_train, x_test = x_train / 255.0, x_test / 255.0 

    # initialize model 
    
    model = LeNet5()

    print(model.summary)

        



