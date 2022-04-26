import cv2
from keras.models import load_model
import numpy as np
import serial
import time
import numpy as np

faceCascade = cv2.CascadeClassifier('Resources/haarcascade_frontalface_default.xml')

ser = serial.Serial('COM7',baudrate=115200,timeout=1)
time.sleep(0.5)
pos = 90
# print(type(pos))

cam = cv2.VideoCapture(0)

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

while True:
    ret, frame = cam.read()
    imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.05, 3)

    for (x, y, w, h) in faces:

        ###############################################################################################
        crop_img = frame[y:y + h, x:x + h]
        img = cv2.resize(crop_img, (224, 224))
        img = img.reshape(1, 224, 224, 3)

        # find out prediction
        # larger value more similar (min:0, max:1)
        prediction = model.predict(img)  # import the webcam image to model for recognition
        print(prediction)

        # find position of each prediction stand for (0~2)
        classIndex = np.argmax(prediction)  # find position of biggest value in array
        print(classIndex)

        # find biggest value of prediction
        probabilityValue = np.amax(prediction)  # find biggest value in array
        print(str(probabilityValue * 100) + "%")

        cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 255, 0), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (80, 255, 0), -2)

        # faces will be detected and namemd when they are trained by model
        if probabilityValue >= 0.75:
            if classIndex == 0 or 1:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 255, 0), 2)
                cv2.rectangle(frame, (x, y - 40), (x + w, y), (80, 255, 0), -2)

                cv2.putText(frame, str(get_className(classIndex)), (x, y - 10), font, 0.75, (255, 255, 255), 1,
                            cv2.LINE_AA)
                cv2.putText(frame, str(round(probabilityValue * 100, 2)) + "%", (10, 110), font, 1.5, (255, 0, 0), 2,
                            cv2.LINE_AA)
        ###############################################################################################

        errorPan = (x + w/2) - 640/2
        print('errorPan', errorPan)
        # print(type(errorPan))
        if abs(errorPan) > 20:
            pos = pos - errorPan/30
            print(type(pos))
        if pos > 160:
            pos = 160
            print("Out of range")
        if pos < 0:
            pos = 0
            print("out of range")
        servoPos = str(pos) + '\r'
        ser.write(servoPos.encode())
        print('servoPos = ', servoPos)
        # print(type(pos))
        time.sleep(0.1)
    cv2.imshow('MBS3523 Webcam', frame)

    if cv2.waitKey(1) & 0xff == 27:
        break

ser.close()
cam.release()
cv2.destroyAllWindows()
