from test import *

# print(get_windows())
print(capture('test2.py — Fuit ninja'))

import cv2 as cv
import mss
import numpy as np
import pyautogui
import time

w, h = pyautogui.size()
img = None
t0 = time.time()
n_frames = 1
monitor = {"top": 0, "left": 0, "width": w, "height": h}
# with mss.mss() as sct:
while True:
    # img = sct.grab(monitor)
    # img = np.array(img)  
    img = capture_full_screen()                      # Convert to NumPy array
    # img = cv.cvtColor(img, cv.COLOR_RGB2BGR)  # Convert RGB to BGR color
    
    small = cv.resize(img, (0, 0), fx=0.5, fy=0.5)
    cv.imshow("Computer Vision", small)

    # Break loop and end test
    key = cv.waitKey(1)
    if key == ord('q'):
        break
    
    elapsed_time = time.time() - t0
    avg_fps = (n_frames / elapsed_time)
    print("Average FPS: " + str(avg_fps))
    n_frames += 1