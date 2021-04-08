# %%
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import numpy as np
from PIL import Image

(train_examples, val_examples), info = tfds.load(
    "cats_vs_dogs",
    with_info=True,
    as_supervised=True,
    split=["train[:80%]", "train[80%:]"],
)

num_examples = info.splits["train"].num_examples
num_classes = info.features["label"].num_classes

IMAGE_RES = 224


def format_image(image, label):
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES)) / 255.0
    return image, label


BATCH_SIZE = 8

train_batches = (
    train_examples.shuffle(num_examples // 4)
    .map(format_image)
    .batch(BATCH_SIZE)
    .prefetch(1)
)
validation_batches = val_examples.map(format_image).batch(BATCH_SIZE).prefetch(1)
images, labels = next(iter(train_batches.take(1)))
images = images.numpy()
labels = labels.numpy()

URL = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"
feature_extractor = hub.KerasLayer(URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))
feature_extractor.trainable = False

model = tf.keras.Sequential([feature_extractor, layers.Dense(2)])

model.summary()

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

EPOCHS = 6
history = model.fit(train_batches, epochs=EPOCHS, validation_data=validation_batches)
# plot stuff
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
plt.show()

# %%
# check predictions
class_names = np.array(info.features["label"].names)
predicted_batch = model.predict(images)
predicted_batch = tf.squeeze(predicted_batch).numpy()
predicted_ids = np.argmax(predicted_batch, axis=-1)
predicted_class_names = class_names[predicted_ids]

print(predicted_class_names)
# %%
print("Labels: ", labels)
print("Predicted labels: ", predicted_ids)
#  some plots
plt.figure(figsize=(10, 9))
for n in range(8):
    plt.subplots(6, 5, n + 1)
    plt.subplots_adjust(hspace=0.3)
    plt.imshow(images[n])
    color = "blue" if predicted_ids[n] == labels else "red"
    plt.title(predicted_class_names[n].title(), color=color)
    plt.axis("off")
    _ = plt.suptitle("Model predictions (blue: correct, red: incorrect)")
