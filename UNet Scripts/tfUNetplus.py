import tensorflow as tf

def conv_block(inputs, num_filters, dropout=0.0):
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    return x

def unet_plus_plus_model(input_shape=(512,512,1), num_classes = 1, deep_supervision=False):
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    #encoding
    x_00 = conv_block(inputs, 64)
    x_10 = conv_block(tf.keras.layers.MaxPooling2D()(x_00), 128)
    x_20 = conv_block(tf.keras.layers.MaxPooling2D()(x_10), 256)
    x_30 = conv_block(tf.keras.layers.MaxPooling2D()(x_20), 512, dropout = 0.3)
    x_40 = conv_block(tf.keras.layers.MaxPooling2D()(x_30), 1024, dropout = 0.3)
    
    #nested decoding pathway
    x_01 = conv_block(tf.keras.layers.concatenate(
        [x_00, tf.keras.layers.UpSampling2D()(x_10)]), 64)
    x_11 = conv_block(tf.keras.layers.concatenate(
        [x_10, tf.keras.layers.UpSampling2D()(x_20)]), 128)
    x_21 = conv_block(tf.keras.layers.concatenate(
        [x_20, tf.keras.layers.UpSampling2D()(x_30)]), 256)
    x_31 = conv_block(tf.keras.layers.concatenate(
        [x_30, tf.keras.layers.UpSampling2D()(x_40)]), 512)

    x_02 = conv_block(tf.keras.layers.concatenate(
        [x_00, x_01, tf.keras.layers.UpSampling2D()(x_11)]), 64)
    x_12 = conv_block(tf.keras.layers.concatenate(
        [x_10, x_11, tf.keras.layers.UpSampling2D()(x_21)]), 128)
    x_22 = conv_block(tf.keras.layers.concatenate(
        [x_20, x_21, tf.keras.layers.UpSampling2D()(x_31)]), 256)

    x_03 = conv_block(tf.keras.layers.concatenate(
        [x_00, x_01, x_02, tf.keras.layers.UpSampling2D()(x_12)]), 64)
    x_13 = conv_block(tf.keras.layers.concatenate(
        [x_10, x_11, x_12, tf.keras.layers.UpSampling2D()(x_22)]), 128)

    x_04 = conv_block(tf.keras.layers.concatenate(
        [x_00, x_01, x_02, x_03, tf.keras.layers.UpSampling2D()(x_13)]), 64)
    
    if deep_supervision:
        outputs = [
            tf.keras.layers.Conv2D(num_classes, 1)(x_01),
            tf.keras.layers.Conv2D(num_classes, 1)(x_02),
            tf.keras.layers.Conv2D(num_classes, 1)(x_03),
            tf.keras.layers.Conv2D(num_classes, 1)(x_04)
        ]
        
        outputs = tf.keras.layers.concatenate(outputs, axis=-1)
        
    else:
        outputs = tf.keras.layers.Conv2D(num_classes, 1)(x_04)
        
        
    
    model = tf.keras.Model(
        inputs = inputs, outputs = outputs, name='Unet_pp_binary')
    
    return model

if __name__ == "__main__":
    model = unet_plus_plus_model(input_shape=(
        512, 512, 1), num_classes=1, deep_supervision=False)
    
    model.summary()
                      
                      
         
        
    