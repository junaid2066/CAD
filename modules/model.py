
"""model.py
Model definitions: simple CNN and transfer-learning wrapper.
"""
import tensorflow as tf
from tensorflow.keras import layers, models, applications

def build_simple_cnn(input_shape=(224,224,3), n_classes=2):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_transfer_model(input_shape=(224,224,3), n_classes=2, base_model_name='resnet50', trainable=False):
    if base_model_name.lower() == 'resnet50':
        base = applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name.lower() == 'mobilenet_v2':
        base = applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError('Unknown base_model_name')
    base.trainable = trainable
    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
