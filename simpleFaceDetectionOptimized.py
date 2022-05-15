import cv2
import detection
import time


cap = cv2.VideoCapture(0)

while cap.isOpened():

    ret, src = cap.read()

    if ret == False:
        break

    start = time.time()

    faces = detection.detectFaces(src, 7, 20)

    end = time.time()
    totalTime = end - start
    fps = 1 / totalTime

    for i in range(len(faces)):
        cv2.rectangle(src, (faces[i][0], faces[i][1]),
                      (faces[i][0] + faces[i][2], faces[i][1] + faces[i][3]),  [0, 0, 255])

    cv2.putText(src, f'FPS: {int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow('detection', src)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
