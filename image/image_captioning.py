import tensorflow
import matplotlib.pyplot as plt

import collections
import random
import numpy as np
import os
import time
import json
from PIL import Image

# download caption files
annotation_folder = "data/annotations"
if not os.path.exists(os.path.abspath(".") + annotation_folder):
    annotation_zip = tf.keras.get_file(
        "captions.zip",
        cache_subdir=os.path.abspath("."),
        origin="http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
        extract=True,
    )
    annotation_file = (
        os.path.dirname(annotation_zip) + "/annotations/captions_train2014.json"
    )
    os.remove(annotation_zip)

# image files
image_folder = "data/train2014"
if not os.path.exists(os.path.abspath(".") + image_folder):
    image_zip = tf.keras.get_file(
        "train2014.zip",
        cache_subdir=os.path.abspath("."),
        origin="http://images.cocodataset.org/zips/train2014.zip",
        extract=True,
    )
    path = os.path.dirname(image_zip) + image_folder
    os.remove(image_zip)
else:
    path = os.path.abspath('.') + image_folder