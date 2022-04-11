import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset',
    labels='inferred',
    label_mode = "int",
    class_names=['1','2','3','4','5','6','7','8','9','10'],
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
    label_mode = "int",
    class_names=['1','2','3','4','5','6','7','8','9','10'],
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
model.summary()

# print(model.layers[0:5])

# for layer in model.layers[0:5]:
#     layer.trainable = False

model.compile(
        loss = tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer = 'adam',
        metrics = ['accuracy']
    )

# train 
model.fit(
    ds_train,
    batch_size=64,
    epochs=100,
    validation_data=ds_test
)

# plot training loss
loss = pd.DataFrame(model.history.history)
loss.plot()
plt.title('Fine Tuning Train/Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Percent')
plt.show()
plt.savefig('Finetuning_Accuracy_Plot')

# test 
model.evaluate(ds_test)

# save model 
model.save('model_finetuned')

# convert and save as tflite model 
converter = tf.lite.TFLiteConverter.from_saved_model('model_finetuned')
tflite_model = converter.convert()
with open('model.tflite','wb') as f:
    f.write(tflite_model)











