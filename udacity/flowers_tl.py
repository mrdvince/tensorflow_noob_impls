import math

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# %%
splits = ["train[:70%]", "train[70%:]"]

(training_set, validation_set), info = tfds.load(
    "tf_flowers", with_info=True, as_supervised=True, split=splits
)
# %%
# some info about dataset
num_classes = info.features["label"].num_classes
num_training_examples = math.ceil(info.splits["train"].num_examples * 0.7)
num_validation_examples = math.floor(info.splits["train"].num_examples * 0.3)
print(f"Total number of classes: {num_classes}")
print(f"Total number of training images: {num_training_examples}")
print(f"Total number of validation images: {num_validation_examples}")

# %%
IMG_RES = 224


def format_image(image, label):
    return tf.image.resize(image, (IMG_RES, IMG_RES)) / 255, label


BATCH_SIZE = 32
train_batches = (
    training_set.shuffle(num_training_examples // 4)
    .map(format_image)
    .batch(BATCH_SIZE)
    .prefetch(1)
)
validation_batches = validation_set.map(format_image).batch(BATCH_SIZE).prefetch(1)
# %%
# feauture extractor (efficientnet)
URL = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"
feature_extractor = hub.KerasLayer(URL, input_shape=(IMG_RES, IMG_RES, 3))
feature_extractor.trainable = False  # freeze
#  attach to a classification head
model = tf.keras.Sequential([feature_extractor, layers.Dense(num_classes)])
model.summary()

# %%
# train model
EPOCHS = 3
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="adam",
    metrics=["accuracy"],
)
# %%
history = model.fit(train_batches, epochs=EPOCHS, validation_data=validation_batches)

# %% plot some metrics
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")

plt.savefig("flowers_tl_loss.png")
# %%
class_names = np.array(info.features["label"].names)
class_names
# %% make predictions on an image batch
image_batch, label_batch = iter(validation_batches).next()

predicted_batch = model.predict(image_batch)
predicted_batch = tf.squeeze(predicted_batch).numpy()

predicted_ids = np.argmax(predicted_batch, axis=-1)
predicted_class_names = class_names[predicted_ids]

# %%
lc = lambda x: class_names[x]
print(f"True labels: \n{lc(label_batch)}, \n\n Predcited labels: \n{lc(predicted_ids)}")
# %% plot model preds
fig = plt.figure(figsize=(25, 12))
for idx in range(BATCH_SIZE):
    ax = fig.add_subplot(4, BATCH_SIZE / 4, idx + 1, xticks=[], yticks=[])
    plt.imshow(np.transpose(image_batch[idx], (0, 1, 2)))
    color = "green" if predicted_ids[idx] == label_batch[idx] else "red"
    ax.set_title(
        f"{predicted_class_names[idx].title(), class_names[label_batch[idx]]}",
        color=color,
    )
    plt.savefig("flowers_tl.png")
# %% save model as pb, load model back as keras model
import time

t = int(time.time())
tf.saved_model.save(model, t)
# load
rm = tf.keras.models.load_model(t, custom_objects={"KerasLayer": hub.KerasLayer})
