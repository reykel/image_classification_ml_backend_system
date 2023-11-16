import os
from io import BytesIO
import numpy as np
import tensorflow as tf
from keras.applications import VGG19
from keras.models import Model
from PIL import Image

img_row = 200
img_col = 200

_weights = './files/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
vgg19 = VGG19(weights=_weights, include_top = False,input_shape = (img_row, img_col,3))

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

def topmodel(bottom_model, num_classes):
    top_model = bottom_model.output
    top_model = tf.keras.layers.Flatten(name='flatten')(top_model)
    top_model = tf.keras.layers.Dense(512, activation='relu')(top_model)
    top_model = tf.keras.layers.Dense(1024, activation = 'relu')(top_model)
    top_model = tf.keras.layers.Dense(512, activation = 'relu')(top_model)
    top_model= tf.keras.layers.Dense(num_classes, activation='softmax')(top_model)
    return top_model

FC_head = topmodel(vgg19, 5),
model = Model(inputs=vgg19.input, outputs=FC_head)

model.compile(optimizer = 'adam',
             loss = 'categorical_crossentropy',
             metrics = ['accuracy'])

model.load_weights('./files/weights.data')

def prediction(image: Image.Image):
    image = np.asarray(image.resize((img_row, img_col)))[..., :3]
    image = np.expand_dims(image, 0)
    image = image / 127.5 - 1.0

    pred = model.predict(image)[0]

    return str(np.argmax(pred))
