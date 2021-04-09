import tensorflow as tf
import tensorflow_datasets as tfds
import math
import numpy as np
import matplotlib.pyplot as plt
import logging

tfds.disable_progress_bar()

logger = tf.get_logger()
logger.setLevel(logging.ERROR)
# %%
dataset, metadata = tfds.load("fashion_mnist", as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset["train"], dataset["test"]
# %%
class_names = metadata.features["label"].names
print(f"Class names {class_names}")
# %%
# explore the data
num_train_samples = metadata.splits["train"].num_examples
num_test_samples = metadata.splits["test"].num_examples
print(f"Number of training samples {num_train_samples}")
print(f"Number of testing samples {num_test_samples}")


# %%

# process the data
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


# map func applies normalize to each element in the dataset
train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)
# cache -> keep in memory making training faster

train_dataset = train_dataset.cache()
test_dataset = test_dataset.cache()

# %%
# explore dataset
for image, label in test_dataset.take(1):
    break
image = image.numpy().reshape((28, 28))

# Plot the image - voila a piece of fashion clothing
plt.figure()
plt.imsave("cloth.png", image, cmap=plt.cm.jet)
# plt.colorbar()
# plt.grid(False)
# plt.savefig("cloth.png",bbox_inches='tight')
# plt.show()
# %%
# plt.figure(figsize=(10,10))
# for i, (image, label) in enumerate(test_dataset.take(25)):
#     image = image.numpy().reshape((28,28))
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(image, cmap=plt.cm.binary)
#     plt.xlabel(class_names[label])
# plt.show()
# %%
# build model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
# %%
# compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
model.summary()
# %%
# train model
BATCH_SIZE = 32
train_dataset = train_dataset.cache().repeat().shuffle(num_train_samples).batch(BATCH_SIZE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE)
# %%
model.fit(train_dataset, epochs=10, steps_per_epoch=math.ceil(num_train_samples / BATCH_SIZE))
# %%%
# evaluate accuracy
test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_samples / BATCH_SIZE))

print(f"Accuracy on test dataset {test_accuracy}")
# %%
# make predictions
for test_image, label in test_dataset.take(1):
    test_image = test_image.numpy()
    label = label.numpy()
    predictions = model.predict(test_image)
# %%
print(predictions.shape)
# %%
print(predictions[0])
print(class_names[np.argmax(predictions[0])])
print(np.argmax(predictions[0]))
print(label[0])
#%%

