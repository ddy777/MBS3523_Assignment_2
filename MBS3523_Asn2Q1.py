#import tensorflow as tf
#from tensorflow import keras
import cv2
from keras.models import load_model
import numpy as np
import time

#import model for detection
facedetect = cv2.CascadeClassifier('Resources/haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)
# cam.set(3, 1280)
# cam.set(4, 960)
font = cv2.FONT_HERSHEY_COMPLEX

#import model for recognition
model = load_model('keras_model.h5')

# variables for FPS
t_old = 0
t_new = 0

def get_className(classNo):
    if classNo == 0:
        return "Myself"
    elif classNo == 1:
        return "Female"
    #elif classNo == 2:
        #return "Female"


while True:
    ret, frame = cam.read()
    faces = facedetect.detectMultiScale(frame, 1.3, 5)

    for x, y, w, h in faces:
        crop_img = frame[y:y + h, x:x + h]
        img = cv2.resize(crop_img, (224, 224))
        img = img.reshape(1, 224, 224, 3)

        #find out prediction
        #larger value more similar (min:0, max:1)
        prediction = model.predict(img) #import the webcam image to model for recognition
        print(prediction)

        #find position of each prediction stand for (0~2)
        classIndex = np.argmax(prediction) #find position of biggest value in array
        print(classIndex)

        #find biggest value of prediction
        probabilityValue = np.amax(prediction) #find biggest value in array
        print(str(probabilityValue * 100) + "%")

        cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 255, 0), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (80, 255, 0), -2)

        #faces will be detected and namemd when they are trained by model
        if probabilityValue >= 0.75:
            if classIndex == 0 or 1:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 255, 0), 2)
                cv2.rectangle(frame, (x, y - 40), (x + w, y), (80, 255, 0), -2)

                cv2.putText(frame, str(get_className(classIndex)), (x, y - 10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, str(round(probabilityValue * 100, 2)) + "%", (10, 110), font, 1.5, (255, 0, 0), 2, cv2.LINE_AA)

        #faces will be detected and named as "Unkown" when they are not trained by model
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 255, 0), 2)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (80, 255, 0), -2)

            cv2.putText(frame, "Unkown", (x, y - 10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, str(round(probabilityValue * 100, 2)) + "%", (10, 110), font, 1.5, (255, 0, 0), 2,
                        cv2.LINE_AA)

    # Calculate FPS and display on upper left
    t_new = time.time()
    fps = 1 / (t_new - t_old)
    t_old = t_new
    cv2.putText(frame, 'FPS = ' + str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    cv2.imshow("Face Recognition Result", frame)
    if cv2.waitKey(1) & 0xff == 27:
        break

cam.release()
cv2.destroyAllWindows()
