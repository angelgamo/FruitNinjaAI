
import cv2 as cv
import mss
import numpy as np
import pyautogui
import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

model = YOLO('best.pt')

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
        # img = np.array(img)
        img = cv.cvtColor(np.array(img), cv.COLOR_BGRA2RGB)

        results = model.predict(img, conf=0.25)

        # for r in results:
        #     annotator = Annotator(img)
        #     boxes = r.boxes
        #     for box in boxes:
        #         b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
        #         c = box.cls
        #         annotator.box_label(b, model.names[int(c)])
        # img = annotator.result()

        for result in results:
            for box in result.boxes:
                left, top, right, bottom = np.array(box.xyxy.cpu(), dtype=np.uint).squeeze()
                width = right - left
                height = bottom - top
                center = (left + int((right-left)/2), top + int((bottom-top)/2))
                label = results[0].names[int(box.cls)]
                confidence = float(box.conf.cpu())

                cv.rectangle(img, (left, top),(right, bottom), (255, 0, 0), 2)

                # cv.putText(img, label,(left, bottom+20),cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv.LINE_AA)

        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
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