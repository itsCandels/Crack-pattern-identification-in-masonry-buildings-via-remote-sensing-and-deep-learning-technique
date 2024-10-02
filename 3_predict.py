#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 11:52:27 2024

@author: federicocandela
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = img / 255.0  # Normalizza l'immagine
    return img

def predict_patch(model, patch):
    patch = np.expand_dims(patch, axis=0)  # Aggiungi una dimensione per il batch
    prediction = model.predict(patch)[0]  # Fai la predizione
    return prediction

def grid_based_prediction(model, image, patch_size=(224, 224)):
    h, w, _ = image.shape
    # Calcoliamo quante righe e colonne ci sono nella griglia
    n_rows = h // patch_size[0]
    n_cols = w // patch_size[1]

    mask_full = np.zeros((h, w), dtype=np.float32)  # Maschera finale vuota

    # Iteriamo attraverso la griglia
    for i in range(n_rows):
        for j in range(n_cols):
            # Estrai il frammento dall'immagine
            patch = image[i * patch_size[0]:(i + 1) * patch_size[0], j * patch_size[1]:(j + 1) * patch_size[1]]

            # Fai la predizione per il frammento
            prediction = predict_patch(model, patch)
            prediction_resized = cv2.resize(prediction, (patch_size[1], patch_size[0]))

            # Posiziona la maschera predetta nel corrispondente punto dell'immagine originale
            mask_full[i * patch_size[0]:(i + 1) * patch_size[0], j * patch_size[1]:(j + 1) * patch_size[1]] = prediction_resized

    return mask_full

def predict_and_visualize(model, image_path, output_folder, patch_size=(224, 224)):
    # Carica e preprocessa l'immagine
    image = load_and_preprocess_image(image_path)

    # Fai la predizione con la griglia
    prediction_full = grid_based_prediction(model, image, patch_size)

    # Statistiche sulla predizione
    print("Statistiche sulla maschera predetta:")
    print("Min:", np.min(prediction_full), "Max:", np.max(prediction_full), "Mean:", np.mean(prediction_full))

    # Scala la predizione tra 0 e 255
    prediction_full = (prediction_full * 255).astype(np.uint8)

    # Carica l'immagine originale per la visualizzazione
    original_image = cv2.imread(image_path)

    # Trova tutti i pixel bianchi nella maschera (consideriamo i pixel con valore > 128)
    red_mask = np.zeros_like(original_image)
    red_mask[prediction_full > 40] = [0, 0, 255]  # Colora i pixel bianchi in rosso

    # Sovrapponi i pixel rossi all'immagine originale per evidenziare la crepa
    overlay = cv2.addWeighted(original_image, 1.0, red_mask, 0.6, 0)

    # Salva l'immagine sovrapposta nel cartella di output
    output_image_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_image_path, overlay)
    print(f"Immagine salvata in: {output_image_path}")

    # Visualizza l'immagine originale, la maschera predetta e la sovrapposizione
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 3, 2)
    plt.title("Predicted Mask")
    plt.imshow(prediction_full, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.show()

if __name__ == "__main__":
    model_path = "output/segmentation_model.h5"
    image_path = "INPUT/prova_a.jpg"
    output_folder = "prediction"

    # Crea la cartella di output se non esiste
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Carica il modello addestrato
    model = load_model(model_path)

    # Fai la predizione e visualizza i risultati
    predict_and_visualize(model, image_path, output_folder)
