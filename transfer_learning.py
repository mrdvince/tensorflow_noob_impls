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

# %%
# Use a tf hub efficient net for classification
CLASSIFIER_URL = "https://tfhub.dev/tensorflow/efficientnet/b0/classification/1"
IMAGE_RES = 224
model = tf.keras.Sequential(
    [hub.KerasLayer(CLASSIFIER_URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))]
)
# %%
# run on a single image
grace_hopper = tf.keras.utils.get_file(
    "image.jpg",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg",
)
grace_hopper = Image.open(grace_hopper).resize((IMAGE_RES, IMAGE_RES))
grace_hopper = np.array(grace_hopper) / 255  # all pixels between 0 and 1
# %%
# Efficient net miss classifiers this, try a mobile net instead
predicted_class = np.argmax(model.predict(grace_hopper[np.newaxis, ...])[0], axis=-1)
labels_path = tf.keras.utils.get_file(
    "ImageNetLabels.txt",
    "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt",
)
imagenet_labels = np.array(open(labels_path).read().splitlines())

plt.imshow(grace_hopper)
plt.axis("off")
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title("Prediction: " + predicted_class_name.title())

# %%
# use the tf hub model for the cats vs dogs dataset
(train_examples, val_examples), info = tfds.load(
    "cats_vs_dogs",
    with_info=True,
    as_supervised=True,
    split=["train[:80%]", "train[80%:]"],
)
num_examples = info.splits["train"].num_examples
num_classes = info.splits["label"].num_classes

# %%
# images not same size, func to make them same size
def format_image(image, label):
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES)) // 255.0
    return image, label


BATCH_SIZE = 8
train_batches = (
    train_examples.shuffle(num_examples // 4)
    .map(format_image)
    .batch(BATCH_SIZE)
    .prefetch(1)
)
validation_batches = val_examples.map(format_image).batch(BATCH_SIZE).prefetch(1)

# %%
# run classifier on a batch of images
images, labels = next(iter(train_batches.take(1)))
images, labels = images.numpy(), labels.numpy()

result_batch = model.predict(images)

predicted_class_names = imagenet_labels[np.argmax(result_batch, axis=-1)]

plt.figure(figsize=(12, 8))
for n in range(30):
    plt.subplot(6, 5, n + 1)
    plt.subplots_adjust(hspace=0.3)
    plt.imshow(images[n])
    plt.title(predicted_class_names[n])
    plt.axis("off")
    _ = plt.suptitle("Imagenet predictions")
