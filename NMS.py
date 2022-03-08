import numpy as np

def NMS(boxes, threshold = 0.4):
    
    #Return an empty list, if no boxex
    if len(boxes) == 0:
        return []
    
    x1 = boxes[:,0] #x coordinate of the top-left corner
    y1 = boxes[:,1] #y coordinate of the top-left corner
    x2 = boxes[:,2] #x coordinate of the bottom-right corner
    y2 = boxes[:,3] #x coordinate of the bottom-right corner
    
    # Compute the area of the bounding boxes and sort the bounding
    # Boxes by the bottom-right y-coordinate of the bounding box
    areas = (x2-x1+1) * (y2-y1+1) # We add 1, because the pixel at the start as well as at the end counts
    
    #The indices of all boxes at start. We will redundant indices one by one.
    indices = np.arange(len(x1))
    
    for i, box in enumerate(boxes):
        
        #Create temporary indices
        temp_indices = indices[indices!=i]
        
        #Find out the coordinates of the inserction box
        xx1 = np.maximum(box[0], boxes[temp_indices,0])
        yy1 = np.maximum(box[1], boxes[temp_indices,1])
        xx2 = np.maximum(box[2], boxes[temp_indices,2])
        yy2 = np.maximum(box[3], boxes[temp_indices,3])
        
        # Find the width and the height of the intersection box
        w = np.maximum(0, xx2-xx1+1)
        h = np.maximum(0, yy2-yy1+1)
        
        # compute the ratio of overlap
        overlap = (w*h)/areas[temp_indices]
        
        #if the actual bounding box has an overlap bigger than treshold with any other bouding box
        
        if np.any(overlap) > threshold:
            indices = indices[indices!=i]
    # return only the boxes at the remaining indices
    return boxes[indices].astype(int)
    