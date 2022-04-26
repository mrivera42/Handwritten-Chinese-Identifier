import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import seaborn as sns

# classes
classes = ['一','二','三','四','五','六','七','八','九','十']

# load model
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset',
    labels='inferred',
    label_mode = "categorical",
    class_names=classes,
    color_mode='grayscale',
    image_size=(28,28),
    shuffle=True,
    seed=123,
    validation_split=0.3,
    subset="training"
)

# load test data 
ds_test = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset',
    labels='inferred',
    label_mode = "categorical",
    class_names=classes,
    color_mode='grayscale',
    image_size=(28,28),
    shuffle=True,
    seed=123,
    validation_split=0.3,
    subset="validation"
)

def normalize(x,y):
    x = tf.cast(x,tf.float32) / 255.0
    return x, y

ds_train = ds_train.map(normalize)
ds_test = ds_test.map(normalize)


model = tf.keras.models.load_model('model_finetuned')
model.evaluate(ds_test)

#error analysis
labels=[]
predictions=[]
for x,y in ds_test:
    labels.append(np.argmax(y,axis=-1))
    predictions.append(model.predict(x).argmax(axis=-1))
print(labels)
print(predictions)

labels_before = []
predictions_before = []
for i in range(0,len(labels)):
    for j in range(0,len(labels[i])):
        labels_before.append(labels[i][j])
        predictions_before.append(predictions[i][j])
print("labels before: ",labels_before)
print("predictions before: ",predictions_before)
cm = tf.math.confusion_matrix(labels=labels_before,predictions=predictions_before)
sns.heatmap(cm,annot=True)
plt.savefig('results/finetuning_cm2.png')


