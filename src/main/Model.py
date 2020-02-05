import tensorflow as tf

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            96, (11, 11), activation="relu", input_shape=(1500, 750, 1)
        ),
        tf.keras.layers.MaxPooling2D(3, 3),
        tf.keras.layers.Conv2D(256, (5, 5), activation="relu"),
        tf.keras.layers.MaxPooling2D(3, 3),
        tf.keras.layers.Dropout(0.01, noise_shape=None, seed=None),
        tf.keras.layers.Conv2D(384, (3, 3), activation="relu"),
        tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(3, 3),
        tf.keras.layers.Dropout(0.01, noise_shape=None, seed=None),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.01, noise_shape=None, seed=None),
        tf.keras.layers.Dense(1, activation="relu"),
    ]
)

