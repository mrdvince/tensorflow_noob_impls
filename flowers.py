# %%
# usual imports
import os
from pathlib import Path
import numpy as np
import glob
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image

# reduce tf logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# %%
_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

zip_file = tf.keras.utils.get_file(origin=_URL, fname="flower_photos.tgz", extract=True)

base_dir = os.path.join(os.path.dirname(zip_file), "flower_photos")

# %%
classes = ["roses", "daisy", "dandelion", "sunflowers", "tulips"]

# %%
# split images into different class folders
for class_ in classes:
    img_path = os.path.join(base_dir, class_)
    images = glob.glob(img_path + "/*.jpg")
    print(f"{class_}: {len(images)} Images")
    train, val = images[: round(len(images) * 0.8)], images[round(len(images) * 0.8) :]

    for t in train:
        Path(os.path.join(base_dir, "train", class_)).mkdir(parents=True, exist_ok=True)
        shutil.move(t, os.path.join(base_dir, "train", class_))

    for v in val:
        Path(os.path.join(base_dir, "val", class_)).mkdir(parents=True, exist_ok=True)
        shutil.move(v, os.path.join(base_dir, "val", class_))
# %%
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
# %%
BATCH_SIZE = 32
IMG_SHAPE = 128
# %%
# plot func
def plot_images(img_array):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(img_array, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.savefig("augs.jpg")
    # plt.show()


# %%
image_gen = image.ImageDataGenerator(
    rescale=1.0 / 255,
    zoom_range=0.5,
    rotation_range=45,
    width_shift_range=0.15,
    height_shift_range=0.15,
)
train_data_gen = image_gen.flow_from_directory(
    train_dir,
    batch_size=BATCH_SIZE,
    target_size=(IMG_SHAPE, IMG_SHAPE),
    shuffle=True,
    class_mode="sparse",
)
# %%
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plot_images(augmented_images)
# %%
# validation dataloader
val_data_gen = image.ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
    val_dir,
    batch_size=BATCH_SIZE,
    target_size=(IMG_SHAPE, IMG_SHAPE),
    class_mode="sparse",
)
# %%
# model
model = tf.keras.models.Sequential(
    [
        layers.Conv2D(16, 3, activation="relu", input_shape=(IMG_SHAPE, IMG_SHAPE, 3)),
        layers.MaxPool2D(2, 2),
        layers.Dropout(0.3),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPool2D(2, 2),
        layers.Dropout(0.3),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPool2D(2, 2),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dense(5, activation="softmax"),
    ]
)
# %%
# compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)
# %%
# train model
EPOCHS = 20
history = model.fit(
    train_data_gen,
    epochs=EPOCHS,
    steps_per_epoch=int(np.ceil(train_data_gen.n / BATCH_SIZE)),
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(val_data_gen.n / BATCH_SIZE)),
)
# %%
#  plot stuff
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()