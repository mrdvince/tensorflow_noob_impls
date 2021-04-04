import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import matplotlib.pyplot as plt
import numpy as np
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

#%%
_URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
zip_dir = tf.keras.utils.get_file(
    "cats_and_dogs_filterted.zip", origin=_URL, extract=True
)

#%%
# list directories
zip_dir_base = os.path.dirname(zip_dir)
# !find $zip_dir_base -type d -print
# %%
base_dir = os.path.join(zip_dir_base, "cats_and_dogs_filtered")

train_cats_dir = os.path.join(os.path.join(base_dir, "train"), "cats")
train_dogs_dir = os.path.join(os.path.join(base_dir, "train"), "dogs")
validation_dogs_dir = os.path.join(os.path.join(base_dir, "validation"), "dogs")
validation_cats_dir = os.path.join(os.path.join(base_dir, "validation"), "cats")
#%%
num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val
# %%
print("total training cat images:", num_cats_tr)
print("total training dog images:", num_dogs_tr)

print("total validation cat images:", num_cats_val)
print("total validation dog images:", num_dogs_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

#%%
BATCH_SIZE = 100
IMG_SHAPE = 128

# %%
img_gen = image.ImageDataGenerator(rescale=1.0 / 255)

train_data_gen = img_gen.flow_from_directory(
    batch_size=BATCH_SIZE,
    directory=os.path.join(base_dir, "train"),
    shuffle=True,
    target_size=(IMG_SHAPE, IMG_SHAPE),
    class_mode="binary",
)
val_data_gen = img_gen.flow_from_directory(
    batch_size=BATCH_SIZE,
    directory=os.path.join(base_dir, "validation"),
    shuffle=False,
    target_size=(IMG_SHAPE, IMG_SHAPE),
    class_mode="binary",
)
# %%
classes = ['cat', 'dog']
sample_imgs, labels = next(train_data_gen)
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2,20/2, idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(sample_imgs[idx], (1,0,2)))
    ax.set_title(classes[int(labels[idx])])
    plt.savefig('cats_dogs.png')
#%%
