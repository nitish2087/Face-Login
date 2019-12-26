import os
import cv2
face_cascade = cv2.CascadeClassifier('/home/nitish/Desktop/project/s_w/opencv-4.1.2/data/haarcascades/haarcascade_frontalface_default.xml')
def face_extractor(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #to convert into gray scale images
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    #return faces
    print(faces)

    if faces is():
        return None

    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

 +   return cropped_face #to crop faces

cam = cv2.VideoCapture(0)

Id = input('enter your id: ')
os.mkdir("/home/nitish/Desktop/project/faces/" + Id)
count = 0

while True:

    ret, frame = cam.read()
    if face_extractor(frame) is not None:
        count += 1
        
        face = cv2.resize(face_extractor(frame), (250, 250))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        file_name_path="/home/nitish/Desktop/project/faces/" + Id + "/" + Id + str(count) + ".jpg"
        cv2.imwrite(file_name_path, face)

        cv2.putText(face, str(count), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Detector', face)
    else:
        print("Face not Found")
        pass

    if cv2.waitKey(1)==13 or count==200:
        break

cam.release()
cv.destroyAllWindows()
print('Collecting Samples Complete!!!')












