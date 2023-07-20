import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import pickle
import time
import pyttsx3

from datetime import datetime, date




path = "C:\\facerecog\\student_image"

images = []
classNames = []
mylist = os.listdir(path)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)
    return encodeList
encoded_face_train = findEncodings(images)
def markAttendance(name):
    today = date.today()
    d1 = today.strftime("%Y-%m-%d")
    with open('Project_Face_Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        already_detected = False
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            if entry[0] == name and entry[1] == d1:
                already_detected = True
                engine.say("face already detected")
                break
        if not already_detected:
            now = datetime.now()
            time_string = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{time_string},{d1}')

# take pictures from webcam 
cap  = cv2.VideoCapture(0)
last_detection_time = time.time()
engine = pyttsx3.init()


# initialize last detection time for all people
last_detection_dict = {name: 0 for name in classNames}

while True:
    current_time = time.time()
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faces_in_frame = face_recognition.face_locations(imgS)
    encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
    for encode_face, faceloc in zip(encoded_faces,faces_in_frame):
        if current_time - last_detection_time > 8:  # add a 8-second delay between detections
            matches = face_recognition.compare_faces(encoded_face_train, encode_face)
            faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
            matchIndex = np.argmin(faceDist)
            print(matchIndex)
            if matches[matchIndex]:
                name = classNames[matchIndex].upper().lower()
                y1,x2,y2,x1 = faceloc
                # since we scaled down by 4 times
                y1, x2,y2,x1 = y1*4,x2*4,y2*4,x1*4

                # check if the face has already been detected today
                if name in last_detection_dict and (current_time - last_detection_dict[name]) < 8:
                    continue


                cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img, (x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                markAttendance(name)
                hour = datetime.now().hour
                if hour < 12:
                    engine.say("Good morning!")
                elif hour < 18:
                    engine.say("Good afternoon!")
                elif hour < 21:
                    engine.say("Good evening!")
                else:
                    engine.say("Good night!")
                engine.say(f"{name}, face detected")
                engine.runAndWait()
                last_detection_dict[name] = current_time
                last_detection_time = current_time
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


