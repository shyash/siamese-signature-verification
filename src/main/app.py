import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img
import glob
from PIL import Image,ImageOps

def showImage(img):
    plt.grid(False)
    plt.gray()
    plt.axis('on')
    plt.imshow(img)
    plt.show() 

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            96, (11, 11), activation="relu", input_shape=(1500, 750, 1),kernel_regularizer=tf.keras.regularizers.l2(2e-4)
        ),
        tf.keras.layers.MaxPooling2D(3, 3),
        tf.keras.layers.Conv2D(256, (5, 5), activation="relu"),
        tf.keras.layers.MaxPooling2D(3, 3),
        tf.keras.layers.Dropout(0.3, noise_shape=None, seed=None),
        tf.keras.layers.Conv2D(384, (3, 3), activation="relu"),
        tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(3, 3),
        tf.keras.layers.Dropout(0.3, noise_shape=None, seed=None),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None)
    ]
)

forged  = [Image.open(im) for im in glob.glob("../../dataset-small/forged/*.png")]
genuine = [Image.open(im) for im in glob.glob("../../dataset-small/genuine/*.png")]
print(genuine)
for i in range(len(genuine)): 
    size = (1500,750)
    genuine[i] = ImageOps.fit(genuine[i], size, Image.BILINEAR) 
    genuine[i] = np.array(genuine[i])
    genuine[i] = np.invert(genuine[i])
    # showImage(genuine[i])
    forged[i] = ImageOps.fit(forged[i], size, Image.BILINEAR) 
    forged[i] = np.array(forged[i])
    forged[i] = np.invert(forged[i]) 
    # showImage(forged[i])