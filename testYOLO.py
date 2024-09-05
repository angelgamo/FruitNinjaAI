import os
import cv2 as cv
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

HOME = os.getcwd()
print(HOME)

model = YOLO(f'{HOME}/best.pt')
results = model.predict(source='screenshot.jpg')

print(results[0].boxes)

detections = results[0].boxes.xyxy

img = cv.imread('screenshot.jpg')

for r in results:
    annotator = Annotator(img)
    boxes = r.boxes
    for box in boxes:
        b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
        c = box.cls
        annotator.box_label(b, model.names[int(c)])
          
img = annotator.result()  
cv.imshow('YOLO V8 Detection', img) 
cv.waitKey(0)