import cv2 as cv
import numpy as np
import time
from windows_capture import WindowsCapture, Frame, InternalCaptureControl

w, h = 1920, 1080
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
            monitor["width"] = fx * 2 - ix
            monitor["height"] = fy * 2 - iy
            cv.setMouseCallback("Computer Vision", lambda *args : None)
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            fx, fy = x, y

capture = WindowsCapture(
    cursor_capture=None,
    draw_border=None,
    monitor_index=None,
    window_name=None,
)

@capture.event
def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
    global img, region, drawing, ix, iy, fx, fy, n_frames, t0, monitor

    img = np.array(frame.frame_buffer)

    if not region:
        small = cv.resize(img, (0, 0), fx=0.5, fy=0.5)
        rectangle = small.copy()
        cv.rectangle(rectangle, (ix, iy), (fx, fy), (0, 255, 0), 1)
        cv.imshow("Computer Vision", rectangle)
    else:
        x0, y0, x1, y1 = monitor['left'],monitor['top'], monitor['left']+monitor['width'],monitor['top']+monitor['height']
        img = img[y0:y1, x0:x1]
        small = cv.resize(img, (0, 0), fx=0.5, fy=0.5)
        cv.imshow("Computer Vision", small)

    key = cv.waitKey(1)
    if key == ord('q'):
        capture_control.stop()

    elapsed_time = time.time() - t0
    avg_fps = (n_frames / elapsed_time)
    print("Average FPS: " + str(avg_fps))
    n_frames += 1

@capture.event
def on_closed():
    print("Capture Session Closed")
    cv2.destroyAllWindows()

cv.namedWindow("Computer Vision")
cv.setMouseCallback("Computer Vision", draw_rect)
capture.start()

print(img.shape)