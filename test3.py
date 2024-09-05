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
region = False
drawing = False
ix, iy = -1, -1
fx, fy = -1, -1

def draw_rect(event, x, y, flags, param):
    global ix, iy, fx, fy, region, drawing
    if event == cv.EVENT_LBUTTONDOWN:
        if not drawing:
            ix, iy = x, y
            fx, fy = x, y
            drawing = True
        else:
            drawing = False
            region = True
            print("Coordenadas: ({}, {}) -> ({}, {})".format(ix, iy, fx, fy))
            # Configura las medidas del monitor aqui
            monitor["top"] = iy
            monitor["left"] = ix
            monitor["width"] = fx - ix
            monitor["height"] = fy - iy
            cv.setMouseCallback("Computer Vision", lambda *args : None)
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            fx, fy = x, y

cv.namedWindow("Computer Vision")
cv.setMouseCallback("Computer Vision", draw_rect)

with mss.mss() as sct:
    img = sct.grab(monitor)
    img = np.array(img)
    small = cv.resize(img, (0, 0), fx=0.5, fy=0.5)
    rectangle = small.copy()

    while not region:
        rectangle = small.copy()
        cv.rectangle(rectangle, (ix, iy), (fx, fy), (0, 255, 0), 1)
        cv.imshow("Computer Vision", rectangle)

        # Break loop and end test
        key = cv.waitKey(1)
        if key == ord('q'):
            break

        elapsed_time = time.time() - t0
        avg_fps = (n_frames / elapsed_time)
        print("Average FPS: " + str(avg_fps))
        n_frames += 1

    t0 = time.time()
    n_frames = 1

    while True:
        img = sct.grab(monitor)
        img = np.array(img)
        
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


print(monitor)