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

# load model
model = tf.keras.models.load_model('model_finetuned')

# recheck accuracy on test data
model.evaluate(ds_test)

#error analysis
labels_with_batches=[]
predictions_with_batches=[]
images_with_batches = []
for x,y in ds_test:
    labels_with_batches.append(np.argmax(y,axis=-1))
    predictions_with_batches.append(model.predict(x).argmax(axis=-1))
    images_with_batches.append(x)
print(labels_with_batches)
print(predictions_with_batches)

labels_no_batches = []
predictions_no_batches = []
images_no_batches = []
for i in range(0,len(labels_with_batches)):
    for j in range(0,len(labels_with_batches[i])):
        labels_no_batches.append(labels_with_batches[i][j])
        predictions_no_batches.append(predictions_with_batches[i][j])
        images_no_batches.append(images_with_batches[i][j])

print("labels before: ",labels_no_batches)
print("predictions before: ",predictions_no_batches)

# confusion matrix
cm = tf.math.confusion_matrix(labels=labels_no_batches,predictions=predictions_no_batches)
sns.heatmap(cm,annot=True)
plt.savefig('results/finetuning_cm2.png')
plt.close()

# look at incorrectly predicted test examples 
indices = np.where(np.array(labels_no_batches) != np.array(predictions_no_batches))
indices = indices[0].tolist()

print("indices",indices)

incorrect = [images_no_batches[i] for i in indices]
count = 0
for i in incorrect:
    index = indices[count]
    plt.imshow(i)
    plt.title(f'Predicted: {predictions_no_batches[index]+1}, Actual: {labels_no_batches[index]+1}')
    plt.savefig(f'results/incorrect/{indices[count]}')
    plt.close()
    count += 1

# error analysis of labels 
sns.displot(data=[labels_no_batches[i] for i in indices])
plt.savefig('results/error_displot.png')

