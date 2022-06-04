import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def load_image_into_numpy_array(path):
    return np.array(Image.open(path))

def predict(img):
  for image_path in img:
    print('Running inference for {}... '.format(image_path), end='')

    image_np=load_image_into_numpy_array(image_path)

    input_tensor=tf.convert_to_tensor(image_np)
    input_tensor=input_tensor[tf.newaxis, ...]
    detections=detect_fn(input_tensor)
    num_detections=int(detections.pop('num_detections'))

    detections={key:value[0,:num_detections].numpy() for key,value in detections.items()}
    detections['num_detections']=num_detections
    detections['detection_classes']=detections['detection_classes'].astype(np.int64)

  image_np_with_detections=image_np.copy()
  viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=100,     
            min_score_thresh=.5,      
            agnostic_mode=False)
  %matplotlib inline
  plt.figure()
  plt.imshow(image_np_with_detections)
  print('Done')
  plt.show()
  
  
#Loading the image
img=['/content/gdrive/MyDrive/Fruit_Detection/workspace/training_demo/images/samples/apple.jpg']
print(img)

predict(img)
