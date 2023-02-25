import cv2
import numpy as np
from imutils.video import VideoStream
from person_detection import Person_Detection
import random
import time
from utils.plots import plot_one_box
from face_identification import load_faceslist, extract_face, inference
import cv2
from facenet_pytorch import MTCNN
import torch
from torchvision import transforms
from PIL import Image
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def handle_left_click(event, x, y, flags, points):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])


def draw_polygon (frame, points):
    for point in points:
        frame = cv2.circle( frame, (point[0], point[1]), 5, (0,0,255), -1)

    frame = cv2.polylines(frame, [np.int32(points)], False, (255,0, 0), thickness=2)
    return frame


def main():
    video = VideoStream(src=0).start()
    
    # Chua cac diem nguoi dung chon de tao da giac
    points = []
    power = pow(10, 6)
    model_person_detection = Person_Detection()
    embeddings, face_names = load_faceslist()
    mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = device)
    detect = False
    run = False
    start = time.time()
    while True:
        frame = video.read()
        frame = cv2.flip(frame, 1)
        temp_frame = frame
        familiar = False
        if detect:
            boxes_face, _ = mtcnn.detect(frame)
            if (time.time() - start) > 5:
                _, det, boxes, indices = model_person_detection.detect(frame=frame, points=points)
                start = time.time()
            run = True

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('d'):
            points.append(points[0])
            detect = True

        # Ve ploygon
        frame_add_polygon = draw_polygon(temp_frame, points)
        if run == True:
            if boxes_face is not None:
                for box in boxes_face:
                    bbox = list(map(int, box.tolist()))
                    face = extract_face(bbox, frame)
                    idx, score = inference(face, embeddings)
                    if idx != -1:
                        familiar = True
                        frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 6)
                        score = torch.Tensor.cpu(score[0]).detach().numpy() * power
                        frame = cv2.putText(frame, face_names[idx] + '_{:.2f}'.format(score), (bbox[0], bbox[1]),
                                            cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 2, cv2.LINE_8)
                    else:
                        frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 6)
                        frame = cv2.putText(frame, 'Unknown', (bbox[0], bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0),
                                            2, cv2.LINE_8)
            # Get names and colors
            names = ['person']
            s = ''
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                if True:  # Add bbox to image
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, frame_add_polygon, label=label, color=(255, 0, 0), line_thickness=1)
                    model_person_detection.draw_prediction(frame_add_polygon, xyxy, points, familiar)

        # # Hien anh ra man hinh
        cv2.imshow("Intrusion Warning", frame_add_polygon)

        cv2.setMouseCallback('Intrusion Warning', handle_left_click, points)

    video.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()