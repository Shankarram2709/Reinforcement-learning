import tensorflow as tf
import numpy as np

keras = tf.keras
K = keras.backend
models = tf.keras.models
layers = tf.keras.layers

class CustomNetV2(object):
        def __init__(self, image_shape, out_channel=3):
                self.image_shape = image_shape
                self.out_channel = out_channel

        def get_model(self):
                inputs = layers.Input(shape=self.image_shape, name='input')
                '''model=tf.keras.applications.mobilenet_v2.MobileNetV2(
                                        include_top=False,
                                        weights=None,
                                        alpha=0.7,
                                        input_tensor=inputs,
                                        pooling=None)'''
                
                model = models.Sequential()

                model.add(layers.Conv2D(32, (5, 5),padding ='same', activation='relu', input_shape=(380,240,1)))
                model.add(layers.MaxPooling2D((2, 2)))

                model.add(layers.Conv2D(64, (5, 5),padding='same', activation='relu'))
                model.add(layers.MaxPooling2D(2, 2))

                model.add(layers.Conv2D(128, (3, 3),padding='same', activation='relu'))
                model.add(layers.MaxPooling2D(2, 2))

                model.add(layers.Conv2D(256, (3, 3),padding='same', activation='relu'))
                model.add(layers.MaxPooling2D(2, 2))

                model.add(layers.Conv2D(512, (3, 3),padding='same', activation='relu'))
                model.add(layers.MaxPooling2D(2, 2))

                model.add(layers.Conv2D(256, (3, 3),padding='same', activation='relu'))
                model.add(layers.MaxPooling2D(2, 2))

                model.add(layers.Conv2D(128, (3, 3),padding='same', activation='relu'))
                model.add(layers.MaxPooling2D(2, 2))

                model.add(layers.Flatten())
                model.add(layers.Dense(512, activation='relu'))
                model.add(layers.Dropout(0.6))
                model.add(layers.Dense(self.out_channel, activation='softmax'))
                #from IPython import embed;embed()
                return model
        


