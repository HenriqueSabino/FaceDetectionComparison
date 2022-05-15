import cv2
import mediapipe as mp
import time

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5) as face_detection:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        start = time.time()

        results = face_detection.process(image)

        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

        cv2.putText(image, f'FPS: {int(fps)}', (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        cv2.imshow('MediaPipe Face Detection', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
