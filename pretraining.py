import tensorflow as tf
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class LeNet5(tf.keras.Model):

    def __init__(self):
        super().__init__()

        self.c1 = tf.keras.layers.Conv2D(
            filters=6,
            kernel_size=(5,5),
            strides=(1,1),
            activation='relu',
            input_shape=(28,28,1))
        self.s2 = tf.keras.layers.AveragePooling2D()
        self.c3 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(5,5),
            strides=(1,1),
            activation='relu')
        self.s4 = tf.keras.layers.AveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.fc5 = tf.keras.layers.Dense(120,activation='relu')
        self.fc6 = tf.keras.layers.Dense(84,activation='relu')
        self.fc7 = tf.keras.layers.Dense(10,activation='softmax')

    def call(self,inputs):

        x = self.c1(inputs)
        x = self.s2(x)
        x = self.c3(x)
        x = self.s4(x)
        x = self.flatten(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)

        return x

    def summary(self):
        x = tf.keras.Input(shape=(28, 28,1))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()


if __name__ == '__main__':
    
    # load dataset 
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    # preprocess 
    x_train, x_test = x_train.reshape(-1,28,28,1) / 255.0, x_test.reshape(-1,28,28,1) / 255.0 
    print(x_train.shape)
    print(x_test.shape)

    # initialize model 
    
    model = LeNet5()
    model.summary()
    model.compile(
        loss = tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer = 'adam',
        metrics = ['accuracy']
    )

    # train 
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=64,
        epochs=40,
        validation_data=(x_test, y_test)
    )

    # plot training loss
    loss = pd.DataFrame(model.history.history)
    loss.plot()
    plt.title('Lenet-5 MNIST Train/Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Percent')
    plt.show()
    plt.savefig('Pretraining_Accuracy_Plot')

    # test 
    model.evaluate(
        x_test, 
        y_test, 
        batch_size=128,
        verbose=2)


    # save model
    model.save('model_pretrained')



    

        



