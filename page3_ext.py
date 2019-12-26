import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('/home/nitish/Desktop/project/s_w/opencv-4.1.2/data/haarcascades/haarcascade_frontalface_alt2.xml')
def face_detector(img, size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)

    if faces is():
        return img,[]

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))

    return img,roi,x,y



labels = {"person-name": 0}

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

with open("lables.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

print(labels)

cap = cv2.VideoCapture(0)

while (True):
    ret, frame = cap.read()

    image,face,x,y = face_detector(frame)
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = recognizer.predict(face)

        if result[1] < 500:
            name = labels[result[0]]
            confidence = int(100*(1-(result[1])/300))
            display_string = name+ str(confidence)+'%'
            cv2.putText(image,display_string,(x,y), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2, cv2.LINE_AA)

        if confidence > 60 :
            cv2.putText(image, "Unlocked", (x,y+260), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Face detector', image)

        else:
            cv2.putText(image,"Locked", (x,y+260), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('Face detector', image)
    except:
        cv2.putText(image, "Face Not Found", (x,y+260), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Face detector', image)
        pass


    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
