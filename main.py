import cv2
import time
import numpy as np
from numpy.core.numeric import ones

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

cap = cv2.VideoCapture(0)
time.sleep(2)

bg = 0

for i in range(60):
    ret, bg = cap.read()

bg = np.flip(bg, axis=1)

while(cap.isOpened()):
    ret, image = cap.read()

    if not ret:
        break
    
    image = np.flip(image, axis=1)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    u_black = np.array([104, 153, 70])
    l_black = np.array([30, 30, 0])
    mask1 = cv2.inRange(hsv, l_black, u_black)

    u_black = np.array([104, 153, 70])
    l_black = np.array([30, 30, 0])
    mask2 = cv2.inRange(hsv, l_black, u_black)
    
    mask1 = mask1 + mask2

    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    mask2 = cv2.bitwise_not(mask1)

    res1 = cv2.bitwise_and(image, image, mask = mask2)
    res2 = cv2.bitwise_and(bg, bg, mask = mask1)

    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    cv2.imshow('C_-_121', final_output)
    cv2.waitKey(1)

cap.release()
output_file.release()
cv2.destroyAllWindows()