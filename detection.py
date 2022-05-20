import cv2
import numpy as np
from deep_sort.tracker import Tracker

net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')

classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

allowed_objects = ["car", "bus"]

count_line_position = 550
detect = []
offset = 6 #allowable error between pixel
counter = 0

def center_handle(x,y,w,h):
     x1 = int(w/2)
     y1 = int(h/2)
     cx = x+x1
     cy = y+y1
     return cx,cy

cap = cv2.VideoCapture("video.mp4")

while True:
    _, img = cap.read()
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255, (608,608),[0,0,0], swapRB=True, crop=False)
    net.setInput(blob)
    layersnames = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(layersnames)

    boxes = []
    confidences = []
    classes_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classes_id = np.argmax(scores)
            confidence = scores[classes_id]

            if confidence > 0.4:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                classes_ids.append(classes_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size = (len(boxes), 3))

    cv2.line(img, (25, count_line_position), (1200,count_line_position),(255,127,0), 3)

    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[classes_ids[i]])
        if label in allowed_objects:
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y+20),font, 2,(255,255,255), 2)

            center = center_handle(x,y,w,h)
            detect.append(center)
            cv2.circle(img, center,4, (0,0,255),-1)

            for (x,y) in detect:
               if y<(count_line_position+offset) and  y>(count_line_position-offset):
                   counter += 1
                   cv2.line(img, (25, count_line_position), (1200,count_line_position),(0,127,255), 3)
                   detect.remove((x,y))

               print("Vehicle Counter:" + str(counter))
               cv2.putText(img, "VEHICLE COUNTER :"+str(counter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)

    cv2.imshow('Image', img)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
