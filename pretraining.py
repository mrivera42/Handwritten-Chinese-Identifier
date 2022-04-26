import tensorflow as tf
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import seaborn as sns

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# class LeNet5(tf.keras.Model):

#     def __init__(self):
#         super().__init__()

#         self.c1 = tf.keras.layers.Conv2D(
#             filters=6,
#             kernel_size=(5,5),
#             strides=(1,1),
#             activation='relu',
#             input_shape=(28,28,1))
#         self.s2 = tf.keras.layers.AveragePooling2D()
#         self.c3 = tf.keras.layers.Conv2D(
#             filters=16,
#             kernel_size=(5,5),
#             strides=(1,1),
#             activation='relu')
#         self.s4 = tf.keras.layers.AveragePooling2D()
#         self.flatten = tf.keras.layers.Flatten()
#         self.fc5 = tf.keras.layers.Dense(120,activation='relu')
#         self.fc6 = tf.keras.layers.Dense(84,activation='relu')
#         self.fc7 = tf.keras.layers.Dense(10,activation='softmax')

#     def call(self,inputs):

#         x = self.c1(inputs)
#         x = self.s2(x)
#         x = self.c3(x)
#         x = self.s4(x)
#         x = self.flatten(x)
#         x = self.fc5(x)
#         x = self.fc6(x)
#         x = self.fc7(x)

#         return x

#     def summary(self):
#         x = tf.keras.Input(shape=(28, 28,1))
#         model = tf.keras.Model(inputs=[x], outputs=self.call(x))
#         return model.summary()


# load dataset 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# preprocess 
x_train = np.array([cv2.bitwise_not(img) for img in x_train])
x_test = np.array([cv2.bitwise_not(img) for img in x_test])
x_train, x_test = x_train.reshape(-1,28,28,1) / 255.0, x_test.reshape(-1,28,28,1) / 255.0 
print(x_train.shape)
print(x_test.shape)

# initialize model (lenet-5)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(
            filters=6,
            kernel_size=(5,5),
            strides=(1,1),
            activation='relu',
            input_shape=(56,56,1)))
model.add(tf.keras.layers.AveragePooling2D())
model.add(tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(5,5),
            strides=(1,1),
            activation='relu'))
model.add(tf.keras.layers.AveragePooling2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(120,activation='relu'))
model.add(tf.keras.layers.Dense(84,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))

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
    epochs=20,
    validation_data=(x_test, y_test)
)

# plot training loss
loss = pd.DataFrame(model.history.history)
fig = loss.plot()
fig.set_title('Pretraining Loss Plot')
fig.set_xlabel('Epoch')
fig.set_ylabel('Percent')
fig = fig.get_figure()
fig.savefig('Pretraining_Loss_Plot')
plt.close()


# test 
model.evaluate(
    x_test, 
    y_test, 
    batch_size=128,
    verbose=2)

labels = y_test
predictions = model.predict(x_test).argmax(axis=-1)
print("labels before: ",labels)
print("predictions before: ",predictions)
cm = tf.math.confusion_matrix(labels=y_test,predictions=predictions)
sns.heatmap(cm,annot=True)
plt.title("Pretraining Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.savefig("pretraining_cm.png")

# save model
model.save('model_pretrained')

# load model 
model2 = tf.keras.models.load_model('model_pretrained')

# error analysis 
model2.evaluate(x_test)
labels = y_test
predictions = model2.predict(x_test).argmax(axis=-1)
print("labels after: ",labels)
print("predictions after: ",predictions)



    

        



