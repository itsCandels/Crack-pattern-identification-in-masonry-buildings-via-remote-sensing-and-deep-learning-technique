#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 16:42:59 2024

@author: federicocandela
"""

import cv2
import numpy as np
import os
import pandas as pd
import random

# Step 1: Process the images to detect cracks, draw bounding boxes, and save results to CSV
def process_crack_image(image_path):
    # Load the crack image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to reduce noise and improve thresholding
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply adaptive thresholding to better isolate the crack
    thresholded_image = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assume the largest contour is the crack
    if contours:
        crack_contour = max(contours, key=cv2.contourArea)
        # Get the bounding box coordinates
        x, y, w, h = cv2.boundingRect(crack_contour)

        return image_rgb, (x, y, x + w, y + h)  # Return image and bounding box
    
    return image_rgb, None

def process_images_and_save_csv(input_folder, csv_path, output_folder=None, num_images=10):
    # Ensure the output folder exists
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.png')]

    # Shuffle and select random images
    random.shuffle(image_files)
    image_files_to_save = image_files[:num_images]

    # Data list to hold bounding box info
    data = []

    # Process each image
    for image_file in image_files_to_save:
        image_path = os.path.join(input_folder, image_file)
        processed_image, bbox = process_crack_image(image_path)
        abs_image_path = os.path.abspath(image_path)

        # Save the processed images to the output folder
        if output_folder:
            abs_output_path = os.path.abspath(os.path.join(output_folder, image_file))
            cv2.imwrite(abs_output_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))

        # Save bounding box to CSV
        if bbox is not None:
            data.append([abs_image_path, *bbox])

    # Save bounding box info to CSV
    df = pd.DataFrame(data, columns=['filepath', 'startX', 'startY', 'endX', 'endY'])
    abs_csv_path = os.path.abspath(csv_path)
    df.to_csv(abs_csv_path, index=False)

    print(f"Processed {num_images} images and saved bounding box info to {abs_csv_path}")

# Step 2: Crop images based on bounding box saved in CSV
def crop_images_from_csv(csv_path, output_folder):
    df = pd.read_csv(csv_path)

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each selected image
    for idx, row in df.iterrows():
        image_path = row['filepath']
        startX, startY, endX, endY = int(row['startX']), int(row['startY']), int(row['endX']), int(row['endY'])

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image {image_path}, skipping.")
            continue

        # Crop the image using bounding box coordinates
        cropped_image = image[startY:endY, startX:endX]

        # Get the filename from the path
        filename = os.path.basename(image_path)

        # Save the cropped image to the output folder
        output_image_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_image_path, cropped_image)

        print(f"Cropped and saved image {output_image_path}")

# Step 3: Generate masks for cropped images
def create_mask_from_cropped_images(cropped_folder, mask_folder):
    # Ensure the mask output folder exists
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)

    # Process each image in the cropped folder
    for image_file in os.listdir(cropped_folder):
        if image_file.endswith('.jpg') or image_file.endswith('.png'):
            image_path = os.path.join(cropped_folder, image_file)

            # Load the cropped image
            image = cv2.imread(image_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur
            blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

            # Apply adaptive thresholding
            thresholded_image = cv2.adaptiveThreshold(
                blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )

            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create a black mask
            mask = np.zeros_like(gray_image)

            # If contours are found, fill the largest contour in white
            if contours:
                crack_contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(mask, [crack_contour], -1, 255, thickness=cv2.FILLED)

            # Save the mask
            output_mask_path = os.path.join(mask_folder, f"mask_{image_file}")
            cv2.imwrite(output_mask_path, mask)

            print(f"Created and saved mask for {image_file}")

# Define paths
input_folder = 'data'
csv_path = 'data_creation/Crack.csv'
cropped_folder = 'data_creation/cropped'
mask_folder = 'data_creation/mask'

# Step 1: Process images, create bounding boxes, and save to CSV
process_images_and_save_csv(input_folder, csv_path, cropped_folder, num_images=1000)

# Step 2: Crop images based on bounding boxes from CSV
crop_images_from_csv(csv_path, cropped_folder)

# Step 3: Create masks from cropped images
create_mask_from_cropped_images(cropped_folder, mask_folder)
