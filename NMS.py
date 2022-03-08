
import pyautogui
import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import copy



def NMS(boxes, overlapThresh = 0.4):
    #return an empty list, if no boxes given
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]  # x coordinate of the top-left corner
    y1 = boxes[:, 1]  # y coordinate of the top-left corner
    x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
    y2 = boxes[:, 3]  # y coordinate of the bottom-right corner
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) # We have a least a box of one pixel, therefore the +1
    indices = np.arange(len(x1))
    for i,box in enumerate(boxes):
        temp_indices = indices[indices!=i]
        xx1 = np.maximum(box[0], boxes[temp_indices,0])
        yy1 = np.maximum(box[1], boxes[temp_indices,1])
        xx2 = np.minimum(box[2], boxes[temp_indices,2])
        yy2 = np.minimum(box[3], boxes[temp_indices,3])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / areas[temp_indices]
        if np.any(overlap) > treshold:
            indices = indices[indices != i]
    return boxes[indices].astype(int)

def bounding_boxes(image, template):
    (tH, tW) = template.shape[:2]             # getting height and width of template 
    imageGray = cv2.cvtColor(image, 0)        # convert the image to grayscale
    templateGray = cv2.cvtColor(template, 0)  # convert the template to grayscale

    result = cv2.matchTemplate(imageGray, templateGray, cv2.TM_CCOEFF_NORMED)  # template matching return the correlatio 
    (y1, x1) = np.where(result >= treshold)  # object is detected, where the correlation is above the treshold
    boxes = np.zeros((len(y1), 4))      # construct array of zeros
    x2 = x1 + tW                       # calculate x2 with the width of the template
    y2 = y1 + tH                       # calculate y2 with the height of the template
    # fill the bounding boxes array
    boxes[:, 0] = x1                 
    boxes[:, 1] = y1
    boxes[:, 2] = x2
    boxes[:, 3] = y2
    return boxes.astype(int)

def draw_bounding_boxes(image,boxes):
    for box in boxes:
        image = cv2.rectangle(copy.deepcopy(image),box[:2], box[2:], (255,0,0), 3)
    return image

if __name__ == "__main__":
    time.sleep(2)
    treshold = 0.8837 # the correlation treshold, in order for an object to be recognised
    template_diamonds = plt.imread(r"/home/kawsar/Desktop/Deep Learning/Object Detection/template.png")

    ace_diamonds_rotated = plt.imread(r"/home/kawsar/Desktop/Deep Learning/Object Detection/image.png")

    boxes_redundant = bounding_boxes(ace_diamonds_rotated, template_diamonds) # calculate bounding boxes
    boxes = NMS(boxes_redundant)                                            # remove redundant bounding boxes
    overlapping_BB_image = draw_bounding_boxes(ace_diamonds_rotated, boxes_redundant)  # draw image with all redundant bounding boxes
    segmented_image = draw_bounding_boxes(ace_diamonds_rotated,boxes)           # draw the bounding boxes onto the image
    plt.imshow(overlapping_BB_image)
    plt.show()
    #plt.imshow(segmented_image)
    #plt.show()
