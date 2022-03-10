import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")
classes = []

with open("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]

outputlayers = net.getUnconnectedOutLayersNames()  

img = cv2.imread("object.jpg")
img = cv2.resize(img, None, fx=0.4,fy=0.4)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

height, widht, channels = img.shape


# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop = False)

class_ids = []
confidences = []
boxes = []

# for b in blob:
#     for n, img_blob in enumerate(b):
#         plt.imshow(img_blob)
#         plt.show()

net.setInput(blob)
outs = net.forward(outputlayers)

# Showing informations on the screen
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            #object detected
            center_x = int(detection[0] * widht)
            center_y = int(detection[1] * height)
            
            w = int(detection[2] * widht)
            h = int(detection[3] * height)
            
            #cv2.circle(img, (center_x, center_y), 10, (0, 255, 0), 2)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            
            #cv2.rectangle(img, (x,y), (x + w, y + h), (0, 255, 0), 2)
            
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
            
#print(len(boxes))
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
font = cv2.LINE_AA

for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(img, label,(x + 10, y + 30), font, .8, (0, 0, 0), 1)


plt.imshow(img)
plt.show()

