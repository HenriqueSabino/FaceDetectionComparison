import cv2
import time

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()

    if ret == False:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    start = time.time()

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    end = time.time()
    totalTime = end - start
    fps = 1 / totalTime

    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.putText(img, f'FPS: {int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow('img', img)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
