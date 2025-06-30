import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask

def load_image(datapoint):
    input_image = tf.image.resize(datapoint['image'], (1024, 1024))
    input_mask = tf.image.resize(
        datapoint['segmentation_mask'],
        (1024,1024),
        method = tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )
    
    input_image, input_mask = normalize(input_image, input_mask)
    
    return input_image, input_mask

class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed = seed)
        self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed = seed)
        
    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels
    

#Using a pretrained MobileNetV2 as encoder

base_model = tf.keras.applications.MobileNetV2(input_shape=[1024, 1024, 3], include_top = False)

layer_names = [
    'block_1_expand_relu',
    'block_3_expand_relu',
    'block_6_expand_relu',
    'block_13_xpand_relu',
    'block_16_project',
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Feature extraction model
down_stack = tf.keras.Model(inputs = base_model.input, outputs = base_model_outputs)
down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels:int):
    inputs = tf.keras.layers.Input(shape=[1024,1024,3])
    
    #Downsampling through model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])
    
    #Upsampling + skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])
        
    #Final Layer
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides = 2,
        padding = 'same')
    
    x = last(x)
    
    return tf.keras.Model(inputs = inputs, outputs = x)
