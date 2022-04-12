import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

classes = ['一','二','三','四','五','六','七','八','九','十']

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset',
    labels='inferred',
    label_mode = "categorical",
    class_names=classes,
    color_mode='grayscale',
    batch_size=64,
    image_size=(28,28),
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="training"
)

ds_test = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset',
    labels='inferred',
    label_mode = "categorical",
    class_names=classes,
    color_mode='grayscale',
    batch_size=64,
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

# def augment(x,y):
#     if tf.random.uniform((),minval=0,maxval=1) < 0.5:
#         image = tf.image.random_brightness(x,max_delta=0.1)
#         image = tf.image.random_contrast(image,lower=0.1,upper=0.2)
#     else:
#         image = x
#     return image, y


# ds_train = ds_train.map(augment)



model = tf.keras.models.load_model("model_pretrained")


# freeze convolutional layers 
for layer in model.layers[0:5]:
    layer.trainable = False
    assert layer.trainable == False
    

model.summary()

model.compile(
        loss = tf.keras.losses.CategoricalCrossentropy(),
        optimizer = 'adam',
        metrics = ['accuracy']
    )

# train 
model.fit(
    ds_train,
    batch_size=64,
    epochs=150,
    validation_data=ds_test
)

# plot training loss
loss = pd.DataFrame(model.history.history)
fig = loss.plot()
fig.set_xlabel('Epoch')
fig.set_ylabel('Percent')
fig.set_title('Fine Tuning Loss Plot')
fig = fig.get_figure()
fig.savefig('Finetuning_Loss_Plot')

# test 
model.evaluate(ds_test)

# # confusion matrix
# test_labels = np.concatenate([y+1 for x, y in ds_test], axis=0)
# test_images = np.concatenate([x for x, y in ds_test], axis=0)

# print("predict classes: ", model.predict(test_images))

# predictions = np.array([np.argmax(model.predict(i),axis=1)+1 for i in test_images])
# print(f'test labels: {test_labels}')
# print(f'predictions: {predictions}')


# cm = confusion_matrix(test_labels, predictions)
# labels = sorted(list(set(test_labels)))
# print(f'labels: {labels}')
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
# disp.plot(cmap=plt.cm.Blues)
# plt.show()
# cm = tf.math.confusion_matrix(test_labels,predictions)




# # save model 
# model.save('model_finetuned')

# # convert and save as tflite model 
# converter = tf.lite.TFLiteConverter.from_saved_model('model_finetuned')
# tflite_model = converter.convert()
# with open('model.tflite','wb') as f:
#     f.write(tflite_model)











