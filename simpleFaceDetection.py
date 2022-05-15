import cv2
import numpy as np
import time


def isSkinPixel(pixel):

    pixel = [int(x) for x in pixel]

    r = pixel[2] / 255.0
    g = pixel[1] / 255.0

    f1 = -1.376 * (r * r) + 1.0743 * r + 0.2
    f2 = -0.776 * (r * r) + 0.5601 * r + 0.18

    w = (r - 0.33) * (r - 0.33) + (g - 0.33) * (g - 0.33)

    a = pixel[2] - pixel[1]
    b = pixel[2] - pixel[0]
    c = pixel[1] - pixel[0]

    value = 0 if (a == 0 and (b == 0 or c == 0)) else 0.5 * \
        (a + b) / np.sqrt(a * a + b * c)

    theta = np.arccos(value)

    H = theta if (pixel[0] <= pixel[1]) else 360 - theta

    return (g > f2 and g < f1 and w > 0.001 and (H <= 20 or H > 240))


def isHairPixel(pixel):

    pixel = [int(x) for x in pixel]

    I = 1.0 / 3 * (pixel[2] + pixel[1] + pixel[0])

    a = pixel[2] - pixel[1]
    b = pixel[2] - pixel[0]
    c = pixel[1] - pixel[0]

    value = 0 if (a == 0 and (b == 0 or c == 0)) else 0.5 * \
        (a + b) / np.sqrt(a * a + b * c)

    theta = np.arccos(value)

    H = theta if (pixel[0] <= pixel[1]) else 360 - theta

    return (I < 80 and (pixel[0] - pixel[1] < 15 or pixel[0] - pixel[2] < 15)) or (H > 20 and H <= 40)


def detectFaces(src, groupWidth, minArea):

    faces = []

    rows = src.shape[0]
    cols = src.shape[1]

    skin = np.zeros((rows // groupWidth,
                    cols // groupWidth, 1), dtype='uint8')
    hair = np.zeros((rows // groupWidth,
                    cols // groupWidth, 1), dtype='uint8')

    detectSkinAndHair(src, skin, hair, groupWidth)

    skinBBox = []
    skinContours, _ = cv2.findContours(
        skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for i in range(len(skinContours)):
        if (cv2.contourArea(skinContours[i]) >= minArea):
            skinBBox.append(cv2.boundingRect(skinContours[i]))

    hairBBox = []
    hairContours, _ = cv2.findContours(hair, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)

    for i in range(len(hairContours)):
        if (cv2.contourArea(hairContours[i]) >= minArea):
            hairBBox.append(cv2.boundingRect(hairContours[i]))

    # Cheking how the boxes overlap
    for i in range(len(skinBBox)):
        for j in range(len(hairBBox)):

            # Matches cases: 0, 1, 2, 3, 5, 6, 7, 9, 11, 12, 15
            if (skinBBox[i][0] >= hairBBox[j][0]):
                # Matches cases: 0, 2, 3, 5, 6, 7, 11, 12, 15
                if (skinBBox[i][1] >= hairBBox[j][1]):
                    # Matches cases: 1
                    if (skinBBox[i][1] + skinBBox[i][3] >= hairBBox[j][1] + hairBBox[j][3]):
                        faces.append(skinBBox[i])
                        hairBBox.remove(hairBBox[j])
                        j -= 1
                        break
                    elif (skinBBox[i][0] + skinBBox[i][2] <= hairBBox[j][0] + hairBBox[j][2]):
                        faces.append(skinBBox[i])
                        hairBBox.remove(hairBBox[j])
                        j -= 1
                        break

                elif (skinBBox[i][0] + skinBBox[i][2] > hairBBox[j][0] + hairBBox[j][2] and
                      skinBBox[i][1] + skinBBox[i][3] > hairBBox[j][1] + hairBBox[j][3]):
                    # Matches cases: 9
                    faces.append(skinBBox[i])
                    hairBBox.remove(hairBBox[j])
                    j -= 1
                    break
            else:
                # Other cases, except 8
                if (skinBBox[i][1] >= hairBBox[j][1]):
                    if (skinBBox[i][1] + skinBBox[i][3] > hairBBox[j][1]):
                        faces.append(skinBBox[i])
                        hairBBox.remove(hairBBox[j])
                        j -= 1
                        break

    # Resizing rects
    for i in range(len(faces)):
        faces[i] = [x * groupWidth for x in faces[i]]

        if (faces[i][0] + faces[i][2] > cols):
            faces[i][2] = cols - faces[i][0]

        if (faces[i][1] + faces[i][3] > rows):
            faces[i][3] = rows - faces[i][1]

    return faces


def detectSkinAndHair(src,  skin,  hair, groupWidth):

    rows = src.shape[0]
    cols = src.shape[1]

    skinPixelCount = 0
    hairPixelCount = 0

    # Going trough every groupWidth'th pixel
    for i in range(0, rows - groupWidth, groupWidth):
        for j in range(0, cols - groupWidth, groupWidth):

            skinPixelCount = 0
            hairPixelCount = 0

            # Running trough a groupWidth x groupWidth pixel grid to determine if the pixel is a skin or hair pixel
            for a in range(i, i + groupWidth):

                # Skipping edge cases
                if (a >= rows):
                    continue

                for b in range(j, j + groupWidth):
                    # Skipping edge cases
                    if (b >= cols):
                        continue

                    color = src[a, b]

                    if (isSkinPixel(color)):
                        skinPixelCount += 1

                    if (isHairPixel(color)):
                        hairPixelCount += 1

            if (skinPixelCount >= (groupWidth * groupWidth) // 2):
                skin[i // groupWidth, j // groupWidth, 0] = 1
            else:
                skin[i // groupWidth, j // groupWidth, 0] = 0

            if (hairPixelCount >= (groupWidth * groupWidth) // 2):
                hair[i // groupWidth, j // groupWidth, 0] = 1
            else:
                hair[i // groupWidth, j // groupWidth, 0] = 0


cap = cv2.VideoCapture(0)

while cap.isOpened():

    ret, src = cap.read()

    if ret == False:
        break

    start = time.time()

    faces = detectFaces(src, 7, 20)

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
