import pickle
import os
import numpy as np
import cv2
import face_recognition
import cvzone
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from datetime import datetime

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{ #this format is used as we're working with JSON format
    'databaseURL': "https://faceattendance-18c5a-default-rtdb.firebaseio.com/",
    'storageBucket': "faceattendance-18c5a.appspot.com"
})

bucket = storage.bucket()

cap = cv2.VideoCapture(0) #0 accesses the laptop's webcam
cap.set(3,640)
cap.set(4,480)

imgBackground = cv2.imread('Resources/background.png') #introducing the background

#Importing the mode images into a list
folderModePath = 'Resources/Modes' #path for the modes
modePathList = os.listdir(folderModePath)
imgModeList = []

for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))
#print(len(imgModeList))

#Load the encoding file
print("Loading Encode File ...")
file = open('EncodeFile.p','rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
#print(studentIds)
print("Encode File loaded")

modeType = 0
counter = 0
id = -1
imgStudent = []
studentInfo = {}

while True:
    success, img = cap.read()
    #Scaling the image down to reduce the computation power by 0.25
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS) #encoding the curret face in the frame
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    imgBackground[162:162+480, 55:55+640] = img #overlaying the background and webcam
    imgBackground[44:44+633, 808:808+414] = imgModeList[modeType] #overlaying the modes

    if faceCurFrame:

        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame): #zip allows us to use the for loop for 2 different lists
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            #print("matches", matches)
            #print("FaceDis", faceDis)

            matchIndex = np.argmin(faceDis)
            #print("Match Index", matchIndex)

            if matches[matchIndex]: #Detecting the Known Faces
                #print("Known Face Detected")
                #print(studentIds[matchIndex]) #prints the id of the student

                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                bbox = 55 + x1, 162 + y1, x2-x1, y2-y1  # bounding box
                imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
                id = studentIds[matchIndex]
                if counter == 0:
                    cvzone.putTextRect(imgBackground, "Loading",(275,400))
                    cv2.imshow("Face Attendance", imgBackground)
                    cv2.waitKey(1)
                    counter = 1
                    modeType = 1

        if counter != 0:

            if counter == 1:
                #Get data
                studentInfo = db.reference(f'Students/{id}').get()  #it gets the info of the student
                print(studentInfo)
                #Get the image from storage
                blob = bucket.get_blob(f'Resized Images/{id}.png') #change or include jpg
                #if blob:
                 #array = np.frombuffer(blob.download_as_string(), np.uint8)
                  #imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
                #else:
                 #print(f"Image file Images/{id}.png not found in the bucket.")
                array = np.frombuffer(blob.download_as_string(), np.uint8)
                imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
                #Update data of attendance
                datetimeObject = datetime.strptime(studentInfo['last_attendance_time'], "%Y-%m-%d %H:%M:%S")
                secondsElapsed = (datetime.now()-datetimeObject).total_seconds()

                print(secondsElapsed)

                if secondsElapsed>=20: #code for "Already Marked" Mode
                    ref = db.reference(f'Students/{id}')
                    studentInfo['total_attendance'] +=1
                    ref.child('total_attendance').set(studentInfo['total_attendance'])
                    ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

                else:
                    modeType = 3
                    counter = 0
                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

            if modeType != 3:

                if 20<counter<30: #showing Marked
                    modeType=2


                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]


                if counter<=20: #Mode 2
                    cv2.putText(imgBackground, str(studentInfo['total_attendance']),(861,125),
                                cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
                    cv2.putText(imgBackground, str(studentInfo['major']), (1006, 550),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(id), (1006, 493),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(studentInfo['standing']), (910, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    cv2.putText(imgBackground, str(studentInfo['year']), (1025, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    cv2.putText(imgBackground, str(studentInfo['starting_year']), (1125, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

                    (w, h), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1) #centering the name text
                    offset = (414-w)//2
                    cv2.putText(imgBackground, str(studentInfo['name']), (808+offset, 445),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

                    imgBackground[175:175+216, 909:909+216] = imgStudent

                counter+=1

                if counter>=30:
                    counter = 0
                    modeType = 0
                    studentInfo = []
                    imgStudent = []
                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    else:
        modeType = 0
        counter = 0

    #cv2.imshow("Webcam", img)
    cv2.imshow("Face Attendance", imgBackground)
    cv2.waitKey(1)

