 import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img
import glob  
from PIL import Image,ImageOps
from sklearn.model_selection import train_test_split
print("Tensorflow version " + tf.__version__) 
def showImage(img):
    plt.grid(False)
    plt.gray()
    plt.axis('off')
    plt.imshow(img)
    plt.show()

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

dataset = []
for i,d in enumerate(glob.glob("../../Dataset/*/")) : 
    dataset.append({
            "forged" : [(np.array(ImageOps.fit(Image.open(im).convert('L'),(1500,750), Image.BILINEAR))/255.0).reshape(1500,750,1) for im in glob.glob(d+"forged/*")],
            "genuine" : [(np.array(ImageOps.fit(Image.open(im).convert('L'),(1500,750), Image.BILINEAR))/255.0).reshape(1500,750,1)  for im in glob.glob(d+"genuine/*")]
        }) 
print(len(dataset))

pairs = []
labels = []
for pt in dataset:    
    anchor = pt["genuine"][0] 
    if(id == 16): print(len(list(zip(pt["forged"],pt["genuine"]))))
    for f,g in zip(pt["forged"],pt["genuine"]):
        pairs.append([anchor,f])
        labels.append(0.0)
        pairs.append([anchor,g])
        labels.append(1.0)
        
print(f"Total Pairs : {len(pairs)}")  
pairs = np.array(pairs) 
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(pairs,labels, test_size=0.33, random_state=42)

def contrastive_loss(y,Dw): 
    alpha = 10**(-4)
    beta = 0.75
    margin = 1
    square_pred = K.square(Dw)
    margin_square = K.square(K.maximum(margin - Dw, 0)) 
    return alpha*(1-y) * square_pred + beta * y * margin_square 

def create_base_network(): 
    input = tf.keras.layers.Input(shape=(1500,750,1))
    x = tf.keras.layers.Conv2D(96, (11, 11), activation="relu", input_shape=(1500, 750,1))(input)
    x = tf.keras.layers.MaxPooling2D(3, 3)(x)
    x = tf.keras.layers.Conv2D(256, (5, 5), activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(3, 3)(x)
    x = tf.keras.layers.Dropout(0.3, noise_shape=None, seed=None)(x)
    x = tf.keras.layers.Conv2D(384, (3, 3), activation="relu")(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(3, 3)(x)
    x = tf.keras.layers.Dropout(0.3, noise_shape=None, seed=None)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None)(x)
     
    return tf.keras.models.Model(input, x)

input_a = tf.keras.layers.Input((1500,750,1))
input_b = tf.keras.layers.Input((1500,750,1))
base_network = create_base_network() 
processed_a = base_network(input_a)
processed_b = base_network(input_b) 

from keras import backend as K 
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

distance = tf.keras.layers.Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([processed_a, processed_b])
model = tf.keras.models.Model([input_a, input_b], distance)
adam = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss=contrastive_loss, optimizer=adam,metrics=['accuracy']) 
model.fit([X_train[:,0], X_train[:,1]], y_train,epochs=10,validation_data=([X_test[:, 0], X_test[:, 1]], y_test))

