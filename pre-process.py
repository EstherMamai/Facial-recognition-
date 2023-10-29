import numpy as np
import cv2
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from PIL import Image


# Specify the path of the folder containing the images
folder_path = 'images'

# Create a new folder to store the preprocessed images
if not os.path.exists('Preprocessed Images'):
    os.makedirs('Preprocessed Images')

# Loop through all the images in the folder
for filename in os.listdir(folder_path):
    image_path = os.path.join(folder_path, filename)

    # Open the image and convert it to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to the grayscale image
    equalized = cv2.equalizeHist(gray)

    # Apply Gaussian smoothing to the equalized image
    smoothed = cv2.GaussianBlur(equalized, (3, 3), 0)

    # Save the preprocessed image in PNG format to the "Preprocessed Images" folder
    new_image_path = os.path.join('Preprocessed Images', filename.replace(".jpg", "").replace(".jpeg", "").replace(".JPG", "").replace(".png", "") + ".png")
    cv2.imwrite(new_image_path, smoothed)

print("Preprocessing Complete")
