import cv2 as cv
import numpy as np
import hand_tracking_module as htm
import os

cap = cv.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 280)
xp, yp = 0, 0

#######################
brush_thickness = 30
eraser_thickness = 60
#######################

overlay = []
imgPath = r"v_p_images/"

for images in os.listdir(imgPath):
    imgpath = os.path.join(imgPath + images)
    bar = cv.imread(imgpath)
    overlay.append(bar)

detector = htm.detector(max_hands=1)
choice = 0
color = (255, 0, 255)

imgcanvas = np.zeros((720, 1280, 3), dtype="uint8")
header = overlay[0]

while True:
    success, frame = cap.read()
    frame = cv.flip(frame, 1)

    # setting the hand tracking
    frame = detector.draw_hands(frame)
    lm_list = detector.find_pos(frame, draw=False)
    fingers = detector.fingers_up()

    if len(lm_list) != 0:
        x, y = lm_list[8][1], lm_list[8][2]
        xp, yp = x, y
        x1, y1 = lm_list[12][1], lm_list[12][2]
        print(fingers)

        # selecting the desired options in option bar
        # 0 - 260 purple , 260 - 517 blue , 517 - 763 , 763 - 990 red , 990 - 1280 eraser
        if fingers[0] and fingers[1]:
            cv.rectangle(frame, (x1 + 15, y1 - 15), (x - 15, y + 15), (255, 255, 255), thickness=cv.FILLED)
            if y < 150:
                if 0 < x < 260:
                    header = overlay[0]
                    color = (255, 0, 255)

                elif 260 < x < 517:
                    header = overlay[1]
                    color = (255, 0, 0)

                elif 517 < x < 763:
                    header = overlay[2]
                    color = (0, 255, 0)

                elif 763 < x < 990:
                    header = overlay[3]
                    color = (0, 0, 255)

                elif 990 < x < 1280:
                    header = overlay[4]
                    color = (0, 0, 0)
        # painting on the screen

        if fingers[0] and not fingers[1]:
            if color == (0, 0, 0):
                cv.circle(frame, (x, y), 30, color, thickness=cv.FILLED)
                cv.line(imgcanvas, (xp, yp), (x, y), color, thickness=eraser_thickness)
            else:
                cv.line(imgcanvas, (xp, yp), (x, y), color, thickness=brush_thickness)

    # setting the header
    frame[0:150, 0:1280] = header

    imgray = cv.cvtColor(imgcanvas, cv.COLOR_BGR2GRAY)
    _, img_inv = cv.threshold(imgray, 50, 255, type=cv.THRESH_BINARY_INV)
    img_inv = cv.cvtColor(img_inv, cv.COLOR_GRAY2BGR)
    frame = cv.bitwise_and(frame, img_inv)
    frame = cv.bitwise_or(frame, imgcanvas)


    cv.imshow("frame", frame)
    if cv.waitKey(1) & 0xFF == ord('d'):
        break
cap.release()
cv.destroyAllWindows()
