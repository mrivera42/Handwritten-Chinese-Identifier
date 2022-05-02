import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import coremltools as ct

classes = ['一','二','三','四','五','六','七','八','九','十']

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset',
    labels='inferred',
    label_mode = "categorical",
    class_names=classes,
    color_mode='grayscale',
    batch_size=1,
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
    batch_size=1,
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
    epochs=50,
    validation_data=ds_test
)

# plot training loss
loss = pd.DataFrame(model.history.history)
fig = loss.plot()
fig.set_xlabel('Epoch')
fig.set_ylabel('Percent')
fig.set_title('Fine Tuning Loss Plot')
fig = fig.get_figure()
fig.savefig('results/Finetuning_Loss_Plot.png')

# test 
print("evaluate before saving")
model.evaluate(ds_test)

# error analysis 
labels=[]
predictions=[]
for x,y in ds_test:
    labels.append(np.argmax(y,axis=-1))
    predictions.append(model.predict(x).argmax(axis=-1))

labels_before = []
predictions_before = []
for i in range(0,len(labels)):
    for j in range(0,len(labels[i])):
        labels_before.append(labels[i][j])
        predictions_before.append(predictions[i][j])
print("labels before: ",labels_before)
print("predictions before: ",predictions_before)
# cm = tf.math.confusion_matrix(labels=labels_before,predictions=predictions_before)
# sns.heatmap(cm,annot=True)
# plt.savefig('results/finetuning_cm.png')

# save model 
model.save('model_finetuned')

# convert to coreml 
# output_labels = classes
# scale = 1/255.

# coreml_model = ct.convert(
#     model,
#     input_names=['image'],
#     image_input_names=['image'],
#     output_names=['output'],
#     class_labels=output_labels,
#     image_scale=scale
# )
# coreml_model.author = 'Max Rivera'
# coreml_model.short_description = 'ML model for handwritten Chinese character classification'
# coreml_model.version = '0.1.0'

# coreml_model.save("CoreModel.mlmodel")

# image_input = ct.ImageType(name="image",shape=(1,28,28,1,),)
# classifier_config = ct.ClassifierConfig(classes)

# mlmodel = ct.convert(
#     model,
#     inputs=[image_input],
#     classifier_config=classifier_config,
# )

# example_image = ds_train[0][0]
# out_dict = mlmodel.predict({"image":example_image})
# print(out_dict["classLabels"])
















