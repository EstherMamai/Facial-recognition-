import pickle
import os
import numpy as np
import cv2
import face_recognition
import cvzone
import firebase_admin
from firebase_admin import credentials, db, storage
from datetime import datetime

def initialize_firebase():
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': "https://faceattendance-18c5a-default-rtdb.firebaseio.com/",
        'storageBucket': "faceattendance-18c5a.appspot.com"
    })
    return storage.bucket()

def load_mode_images(folderPath):
    modePathList = os.listdir(folderPath)
    return [cv2.imread(os.path.join(folderPath, path)) for path in modePathList]

def load_encoded_faces(filePath):
    with open(filePath, 'rb') as file:
        return pickle.load(file)

def classify_face(frame, encodeListKnown, studentIds):
    faceCurFrame = face_recognition.face_locations(frame)
    encodeCurFrame = face_recognition.face_encodings(frame, faceCurFrame)
    results = []
    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            results.append((studentIds[matchIndex], faceLoc))
    return results

def update_attendance(id, studentInfo, bucket):
    ref = db.reference(f'Students/{id}')
    studentInfo = ref.get()
    blob = bucket.get_blob(f'Resized Images/{id}.png')
    if not blob:
        return None
    array = np.frombuffer(blob.download_as_string(), np.uint8)
    imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
    
    datetimeObject = datetime.strptime(studentInfo['last_attendance_time'], "%Y-%m-%d %H:%M:%S")
    secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
    
    if secondsElapsed >= 20:
        studentInfo['total_attendance'] += 1
        ref.child('total_attendance').set(studentInfo['total_attendance'])
        ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        return imgStudent
    return None

def main():
    bucket = initialize_firebase()
    imgModeList = load_mode_images('Resources/Modes')
    encodeListKnown, studentIds = load_encoded_faces('EncodeFile.p')

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    imgBackground = cv2.imread('Resources/background.png')

    modeType = 0
    counter = 0
    id = -1
    imgStudent = []

    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        results = classify_face(imgS, encodeListKnown, studentIds)
        imgBackground[162:162+480, 55:55+640] = img
        imgBackground[44:44+633, 808:808+414] = imgModeList[modeType]

        if results:
            id, faceLoc = results[0]
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
            imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)

            if counter == 0:
                cvzone.putTextRect(imgBackground, "Loading", (275, 400))
                cv2.imshow("Face Attendance", imgBackground)
                cv2.waitKey(1)
                counter = 1
                modeType = 1

            if counter != 0:
                imgStudent = update_attendance(id, {}, bucket)
                if imgStudent is not None:
                    if counter == 1:
                        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
                    if 20 < counter < 30:
                        modeType = 2
                    if counter <= 20:
                        # Display student information
                        # (omitted for brevity)
                        imgBackground[175:175+216, 909:909+216] = imgStudent
                    counter += 1
                    if counter >= 30:
                        counter = 0
                        modeType = 0
                        imgStudent = []

        else:
            modeType = 0
            counter = 0

        cv2.imshow("Face Attendance", imgBackground)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()