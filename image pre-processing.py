import cv2
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from PIL import Image


cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{ #this format is used as we're working with JSON format
    'databaseURL':"https://faceattendance-18c5a-default-rtdb.firebaseio.com/",
    'storageBucket': "faceattendance-18c5a.appspot.com",
})


# Specify the path of the folder containing the images
folder_path = 'images'

# Create a new folder to store the resized images
if not os.path.exists('Resized Images'):
    os.makedirs('Resized Images')

# Loop through all the images in the folder
for filename in os.listdir(folder_path):
    image_path = os.path.join(folder_path, filename)

    # Open the image and resize it to 216x216
    with Image.open(image_path) as img:
        img = img.resize((216, 216))


        # Save the image in PNG format to the "Resized Images" folder
        new_image_path = os.path.join('Resized Images', filename.replace(".jpg", "").replace(".jpeg", "").replace(".JPG", "").replace(".png", "") + ".png")
        img.save(new_image_path, "PNG")

print("Resized Complete")
