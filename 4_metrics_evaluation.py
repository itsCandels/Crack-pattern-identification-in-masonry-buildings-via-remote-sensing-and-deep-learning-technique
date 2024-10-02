#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 11:05:14 2024

@author: federicocandela
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(image_folder, mask_folder, target_size=(224, 224)):
    images = []
    masks = []
    image_paths = sorted([os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith(('.jpg', '.jpeg', '.png'))])
    mask_paths = sorted([os.path.join(mask_folder, fname) for fname in os.listdir(mask_folder) if fname.endswith(('.jpg', '.jpeg', '.png'))])
    
    for img_path, mask_path in zip(image_paths, mask_paths):
        # Carica l'immagine
        img = cv2.imread(img_path)
        img = cv2.resize(img, target_size)
        img = img / 255.0  # Normalizza l'immagine tra 0 e 1

        # Carica la maschera in scala di grigi e binarizzala
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, target_size)
        mask = (mask > 127).astype(np.uint8)  # Binarizza le maschere con soglia 127

        images.append(img)
        masks.append(mask)
    
    return np.array(images), np.array(masks)

def calculate_metrics(y_true, y_pred, threshold=0.5):
    # Binarizza sia la maschera vera che la previsione
    y_true = (y_true > threshold).astype(np.uint8)
    y_pred = (y_pred > threshold).astype(np.uint8)
    
    # Flatten dei valori veri e predetti
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Calcolo delle metriche
    iou = jaccard_score(y_true_flat, y_pred_flat, average='binary')
    dice = f1_score(y_true_flat, y_pred_flat, average='binary')
    precision = precision_score(y_true_flat, y_pred_flat, average='binary')
    recall = recall_score(y_true_flat, y_pred_flat, average='binary')
    accuracy = accuracy_score(y_true_flat, y_pred_flat)

    # Calcolo matrice di confusione
    conf_matrix = confusion_matrix(y_true_flat, y_pred_flat)

    return iou, dice, precision, recall, accuracy, conf_matrix

def evaluate_thresholds(y_true, y_pred, thresholds):
    best_threshold = 0.2  # Soglia di default
    best_dice = 0  # Partiamo da un valore minimo
    metrics = {
        "iou": [],
        "dice": [],
        "precision": [],
        "recall": [],
        "accuracy": []
    }

    for threshold in thresholds:
        iou, dice, precision, recall, accuracy, _ = calculate_metrics(y_true, y_pred, threshold=threshold)
        metrics["iou"].append(iou)
        metrics["dice"].append(dice)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["accuracy"].append(accuracy)
        
        # Controlla se questo threshold massimizza il Dice score
        if dice > best_dice:
            best_dice = dice
            best_threshold = threshold

    return best_threshold, metrics

def evaluate_model(model_path, image_folder, mask_folder, output_folder, target_size=(224, 224), thresholds=np.arange(0.1, 0.9, 0.1)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Carica modello
    model = load_model(model_path)

    # Carica dati di test
    images, masks = load_data(image_folder, mask_folder, target_size=target_size)

    # Previsioni del modello
    predictions = model.predict(images)

    iou_scores = []
    dice_scores = []
    precision_scores = []
    recall_scores = []
    accuracy_scores = []
    confusion_matrices = []

    # Trova la soglia ottimale per ogni immagine
    for i in range(len(predictions)):
        best_threshold, metrics_results = evaluate_thresholds(masks[i], predictions[i], thresholds)
        iou, dice, precision, recall, accuracy, conf_matrix = calculate_metrics(masks[i], predictions[i], threshold=best_threshold)
        iou_scores.append(iou)
        dice_scores.append(dice)
        precision_scores.append(precision)
        recall_scores.append(recall)
        accuracy_scores.append(accuracy)
        confusion_matrices.append(conf_matrix)

    avg_iou = np.mean(iou_scores)
    avg_dice = np.mean(dice_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_accuracy = np.mean(accuracy_scores)

    # Salva risultati
    results_path = os.path.join(output_folder, 'evaluation_metrics.txt')
    with open(results_path, 'w') as f:
        f.write(f"Average IoU: {avg_iou:.4f}\n")
        f.write(f"Average Dice Coefficient (F1 Score): {avg_dice:.4f}\n")
        f.write(f"Average Precision: {avg_precision:.4f}\n")
        f.write(f"Average Recall: {avg_recall:.4f}\n")
        f.write(f"Average Accuracy: {avg_accuracy:.4f}\n")
        f.write("\nPer-image metrics:\n")
        for i, (iou, dice, precision, recall, accuracy) in enumerate(zip(iou_scores, dice_scores, precision_scores, recall_scores, accuracy_scores)):
            f.write(f"Image {i+1} - IoU: {iou:.4f}, Dice: {dice:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}\n")

    # Genera grafici
    plt.figure()
    plt.plot(iou_scores, label="IoU")
    plt.plot(dice_scores, label="Dice (F1 Score)")
    plt.plot(precision_scores, label="Precision")
    plt.plot(recall_scores, label="Recall")
    plt.plot(accuracy_scores, label="Accuracy")
    plt.title("Evaluation Metrics per Image")
    plt.xlabel("Image #")
    plt.ylabel("Score")
    plt.legend()
    plot_save_path = os.path.join(output_folder, "metrics_plot.png")
    plt.savefig(plot_save_path)
    plt.show()

    # Genera la matrice di confusione aggregata
    aggregated_conf_matrix = np.sum(confusion_matrices, axis=0)
    plt.figure(figsize=(10, 7))
    sns.heatmap(aggregated_conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Aggregated Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(os.path.join(output_folder, "confusion_matrix.png"))
    plt.show()

if __name__ == "__main__":
    model_path = "output/segmentation_model.h5"
    image_folder = "/Users/federicocandela/Desktop/GIT_HUB/2_U_NET_CRACK_DETECTION/dataset/val/images"
    mask_folder = "/Users/federicocandela/Desktop/GIT_HUB/2_U_NET_CRACK_DETECTION/dataset/val/masks"
    output_folder = "evaluation"

    evaluate_model(model_path, image_folder, mask_folder, output_folder)
