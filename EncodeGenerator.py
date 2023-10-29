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


# # Specify the path of the folder containing the images
# folder_path = 'images'
#
# # Create a new folder to store the resized images
# if not os.path.exists('Resized Images'):
#     os.makedirs('Resized Images')
#
# # Loop through all the images in the folder
# for filename in os.listdir(folder_path):
#     image_path = os.path.join(folder_path, filename)
#
#     # Open the image and resize it to 216x216
#     with Image.open(image_path) as img:
#         img = img.resize((216, 216))
#
#
#         # Save the image in PNG format to the "Resized Images" folder
#         new_image_path = os.path.join('Resized Images', filename.replace(".jpg", "").replace(".jpeg", "").replace(".JPG", "").replace(".png", "") + ".png")
#         img.save(new_image_path, "PNG")
#
# print("Resized Complete")

#Importing student images
folderPath = 'Preprocessed Images' #path for the resized images
pathList = os.listdir(folderPath)
print(pathList)
imgList = []
studentIds = []
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    #print(imgList)
    print(path)
    print(os.path.splitext(path)[0]) #obtaining the ids from the file name
    studentIds.append(os.path.splitext(path)[0])

    fileName = f'{folderPath}/{path}'#creates a folder called images in storage
    bucket = storage.bucket()
    blob = bucket.blob(fileName) #for sending
    blob.upload_from_filename(fileName)



print(studentIds)

def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        # openCV uses BGR but facial recognition uses RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #find encodings
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode) #Loops through to save all encodings of the images

    return encodeList


print("Encoding Started ...") #takes a while if there are a lot of images
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, studentIds] #the two lists to be stored in the pickle file
print(encodeListKnownWithIds) #prints the encodings of the images
print("Encoding Complete")

#storing the encodings with Ids in the pickle file so that we can import it while we're using the webcam
file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("File Saved")











