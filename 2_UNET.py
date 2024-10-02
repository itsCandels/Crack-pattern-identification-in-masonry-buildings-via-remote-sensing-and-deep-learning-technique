#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 11:19:06 2024

@author: federicocandela
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

def unet_model(input_size=(224, 224, 3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model

def load_data(image_folder, mask_folder, target_size=(224, 224)):
    images = []
    masks = []
    image_paths = sorted([os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith(('.jpg', '.jpeg', '.png'))])
    mask_paths = sorted([os.path.join(mask_folder, fname) for fname in os.listdir(mask_folder) if fname.endswith(('.jpg', '.jpeg', '.png'))])
    
    for img_path, mask_path in zip(image_paths, mask_paths):
        img = load_img(img_path, target_size=target_size)
        img = img_to_array(img)
        img = img / 255.0  # Normalizza l'immagine
        
        mask = load_img(mask_path, target_size=target_size, color_mode="grayscale")
        mask = img_to_array(mask)
        mask = mask / 255.0  # Normalizza la maschera
        
        images.append(img)
        masks.append(mask)
    
    return np.array(images), np.array(masks)

def train_unet(image_folder, mask_folder, output_folder, batch_size=16, epochs=50, lr=1e-4):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images, masks = load_data(image_folder, mask_folder)
    (trainImages, testImages, trainMasks, testMasks) = train_test_split(images, masks, test_size=0.20, random_state=42)

    model = unet_model(input_size=(224, 224, 3))
    opt = Adam(learning_rate=lr)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    H = model.fit(trainImages, trainMasks,
                  validation_data=(testImages, testMasks),
                  batch_size=batch_size,
                  epochs=epochs, verbose=1, callbacks=[reduce_lr, early_stopping])

    model_save_path = os.path.join(output_folder, "segmentation_model.h5")
    model.save(model_save_path)

    history_save_path = os.path.join(output_folder, "training_history.pkl")
    with open(history_save_path, 'wb') as f:
        pickle.dump(H.history, f)

    plt.figure()
    plt.plot(H.history["loss"], label="train_loss")
    plt.plot(H.history["val_loss"], label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plot_save_path = os.path.join(output_folder, "training_loss_plot.png")
    plt.savefig(plot_save_path)
    plt.show()

if __name__ == "__main__":
    image_folder = "dataset/train/images"
    mask_folder = "dataset/train/images"
    output_folder = "output"
    train_unet(image_folder, mask_folder, output_folder)

