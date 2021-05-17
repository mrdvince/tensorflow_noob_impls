import tensorflow as tf

# basic keras cnn impl
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(2, 2), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size="2,2"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)
