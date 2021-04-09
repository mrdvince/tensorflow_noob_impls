import tensorflow as tf
import numpy as np
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# training data
celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

for i, c in enumerate(celsius_q):
    print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))

# create model
fc0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([fc0])
# model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])

# compile model
model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(0.1))

#  train model

history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training model")

# plot stats

import matplotlib.pyplot as plt

plt.xlabel("Epoch Number")
plt.ylabel("Loss Magnitude")
plt.plot(history.history["loss"])

# use model to predict values

print(model.predict([100.0]))

# layer weights
print(fc0.get_weights())

# more layers
fc0 = tf.keras.layers.Dense(units=4, input_shape=[1])
fc1 = tf.keras.layers.Dense(units=4)
fc2 = tf.keras.layers.Dense(units=1)

model2 = tf.keras.Sequential([fc0, fc1, fc2])
model2.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(0.1))
model2.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")
print(model.predict([100.0]))
print(
    "Model predicts that 100 degrees Celsius is: {} degrees Fahrenheit".format(
        model.predict([100.0])
    )
)
print("These are the l0 variables: {}".format(fc0.get_weights()))
print("These are the l1 variables: {}".format(fc1.get_weights()))
print("These are the l2 variables: {}".format(fc2.get_weights()))
