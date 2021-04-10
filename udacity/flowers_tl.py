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