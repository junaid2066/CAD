
"""train.py
Training loop using Keras.
"""
import os
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from modules.model import build_transfer_model

def train_model(X_train, y_train, X_val, y_val, batch_size=32, epochs=20, out_dir='checkpoints', model_name='model.h5'):
    os.makedirs(out_dir, exist_ok=True)
    input_shape = X_train.shape[1:]
    model = build_transfer_model(input_shape=input_shape, n_classes=len(set(y_train)))
    checkpoint = ModelCheckpoint(os.path.join(out_dir, model_name), save_best_only=True, monitor='val_loss')
    early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    train_gen = datagen.flow(X_train, y_train, batch_size=batch_size)
    history = model.fit(train_gen, validation_data=(X_val, y_val), epochs=epochs, callbacks=[checkpoint, early])
    return model, history
