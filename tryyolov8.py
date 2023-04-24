import torch
import cv2
from ultralytics import YOLO
import numpy as np

def processoneresult(result, img):
    # cpu is a pytorch tensor method
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype='int')
    classes = np.array(result.boxes.cls.cpu(), dtype='int')
    confidences = np.array(result.boxes.conf.cpu(), dtype='float')
    # segmentations = np.array(result.masks.xy.cpu(), dtype='int')
    color_list = [(0, 0, 255),
                  (0, 255, 0),
                  (255, 0, 0),
                  (255, 0, 255),
                  (0, 255, 255),
                  (255, 255, 0),
                  (128, 128, 128)]
    for cls, box, conf in zip(classes, bboxes, confidences):
        (x, y, x2, y2) = box
        cv2.rectangle(img, (x, y), (x2, y2), color_list[cls % 7], 1)
        cv2.putText(img, str(result.names[cls]), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color_list[cls % 7], 2)
        cv2.putText(img, "{:.2f}".format(conf), (x, y2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color_list[cls % 7], 2)

if __name__ == "__main__":
    # load a image
    img = cv2.imread('generate/frame00113.jpg')
    model = YOLO('yolov8x')
    # yolo results are pytorch tensors
    results = model(img)
    for result in results:
        processoneresult(result, img)
    cv2.imshow('img', img)
    cv2.waitKey(0)