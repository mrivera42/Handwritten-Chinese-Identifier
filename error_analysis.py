import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# classes
classes = ['一','二','三','四','五','六','七','八','九','十']


ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset',
    labels='inferred',
    label_mode = "categorical",
    class_names=classes,
    color_mode='grayscale',
    image_size=(28,28),
    shuffle=True,
    batch_size=1,
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
    batch_size=1,
    seed=123,
    validation_split=0.3,
    subset="validation"
)

def normalize(x,y):
    x = tf.cast(x,tf.float32) / 255.0
    return x, y

ds_train = ds_train.map(normalize)
ds_test = ds_test.map(normalize)

# load model
model = tf.keras.models.load_model('model_finetuned')

# recheck accuracy on test data
model.evaluate(ds_test)

# get labels and predictions 
labels = []
predictions = []
images = []
for x,y in ds_test:
    labels.append(classes[np.argmax(y,axis=-1)[0]])
    predictions.append(classes[model.predict(x).argmax(axis=-1)[0]])
    images.append(x)
print(f'labels: {labels}')
print(f'predictions: {predictions}')

# confusion matrix 
cm = confusion_matrix(labels,predictions)
sns.heatmap(cm,annot=True)
plt.savefig('results/finetuning_cm2.png')
plt.close()

# look at incorrectly predicted test examples 
indices = np.where(np.array(labels) != np.array(predictions))[0].tolist()
print(f'indices: {indices}')

incorrect = [images[i] for i in indices]
count = 0
for i in incorrect:
    index = indices[count]
    plt.imshow(i)
    plt.title(f'Predicted: {predictions[index]+1}, Actual: {labels[index]+1}')
    plt.savefig(f'results/incorrect/{indices[count]}')
    plt.close()
    count += 1

# error analysis of labels 
sns.displot(data=[labels[i] for i in indices])
plt.savefig('results/error_displot.png')




