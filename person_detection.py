from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import cv2
import numpy as np
import os
import sys
from pathlib import Path
import torch
from telegram_utils import flow_send_telegram
import datetime
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, \
    scale_coords
from utils.torch_utils import select_device
from models.experimental import attempt_load
from settings import MODEL_PERSON_DETECTION


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = attempt_load(MODEL_PERSON_DETECTION, map_location=device)  # load FP32 model


def isInside(points, centroid):
    polygon = Polygon(points)
    centroid = Point(centroid)
    print(polygon.contains(centroid))
    return polygon.contains(centroid)


class Person_Detection():
    def __init__(self, detect_class="person", frame_width=1280, frame_height=720):
        # Parameters
        self.classes = ['person']
        self.nms_threshold = 0.4
        self.imgsz = (416, 416)  # inference size (height, width)
        self.conf_thres = 0.70  # confidence threshold
        self.iou_thres = 0.45
        self.detect_class = detect_class
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.scale = 1 / 255
        self.model = model
        # Load model
        self.output_layers = None
        self.last_alert = None
        self.is_bbox_null = True
        self.alert_telegram_each = 45  # seconds


    def draw_prediction(self, img, x , points, familiar):
        color = (0, 0, 255)
        # # Tinh toan centroid

        centroid = ((int(x[0]) + int(x[2])) // 2, (int(x[1])+ int(x[3])) // 2)
        # cv2.circle(img, centroid, 5, (color), -1)

        if isInside(points, centroid) and familiar == False:
            img = self.alert(img)

        return isInside(points, centroid)

    def alert(self, img):
        cv2.putText(img, "ALARM!!!!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # New thread to send telegram after 15 seconds
        if (self.last_alert is None) or (
                (datetime.datetime.utcnow() - self.last_alert).total_seconds() > self.alert_telegram_each) or \
                self.is_bbox_null == True:
            self.is_bbox_null = False
            self.last_alert = datetime.datetime.utcnow()
            print("Send Messenger!!!")
            cv2.imwrite("alert.png", img)
            flow_send_telegram()
        return img

    def detect_person(
            self,
            source,
            model,
            imgsz = 416,
            device='cpu',
            conf_thres=0.70,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
    ):
        device = select_device(device)
        half = device.type != 'cpu'  # half precision only supported on CUDA
        # Load model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size

        img0 = source
        img = letterbox(img0, imgsz, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s = ''
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        return img0, det

    def detect(self, frame, points):
        # Loc cac object trong khung hinh
        class_ids = []
        confidences = []
        boxes = []
        # Inference
        img, outs = self.detect_person(frame, model = self.model)

        for detection in outs:
            class_id = int(detection[5])
            confidence = detection[4]
            if (float(confidence) >= self.conf_thres) and (self.classes[class_id] == self.detect_class):
                center_x = int(detection[0])
                center_y = int(detection[1])
                w = int(detection[2])
                h = int(detection[3])
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
        if len(boxes) == 0:
            self.is_bbox_null = True
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thres, self.nms_threshold)

        # for i in indices:
        #     box = boxes[i]
        #     x = box[0]
        #     y = box[1]
        #     w = box[2]
        #     h = box[3]
        #     self.draw_prediction(frame, class_ids[i], round(x), round(y), round(w), round(h), points)

        return frame, outs, boxes, indices
