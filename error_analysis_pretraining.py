import tensorflow as tf 
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.array([cv2.bitwise_not(img) for img in x_train])
x_test = np.array([cv2.bitwise_not(img) for img in x_test])
x_train, y_train = x_train / 255.0, y_train / 255.0
model = tf.keras.models.load_model('model_pretrained')

# error analysis 
model.evaluate(x_test)
labels = y_test
predictions = model.predict(x_test).argmax(axis=-1)
print(labels)
print(predictions)
cm = tf.math.confusion_matrix(labels=y_test,predictions=predictions)
sns.heatmap(cm,annot=True)
plt.show()
