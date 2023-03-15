import numpy as np
import cv2
from imutils.video import VideoStream
import imutils
import time

prototxt_path = "./MobileNetSSD_deploy_prototxt.txt"
model_path = "./MobileNetSSD_deploy.caffemodel"

conf_limit = 0.25

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "window"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("Loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

video_path = None
if video_path is None:
    vs = VideoStream().start()
    # warm up the camera
    time.sleep(2)
else:
    vs = cv2.VideoCapture(video_path)

while True:
    vid_frame = vs.read()
    (h, w) = vid_frame.shape[:2]
    vblob = cv2.dnn.blobFromImage(cv2.resize(
        vid_frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(vblob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):  # extract the confidence
        confidence = detections[0, 0, i, 2]
        if confidence > conf_limit:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            # print("{}".format(label))
            cv2.rectangle(vid_frame, (startX, startY),
                          (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(vid_frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    if vid_frame is None:
        break

    vid_frame = imutils.resize(vid_frame, width=800)
    cv2.imshow("Camera image", vid_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
