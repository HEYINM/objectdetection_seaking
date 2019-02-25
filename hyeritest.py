#Copyed and modify to read a video.avi instead of the .jpg files from Note need OpenCV installed also
#https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
#from this repository
#https://github.com/tensorflow/models/tree/master/research/object_detection

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
import copy
#import cv2 #Use OpenCV instead of Matplot

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import math
# This is needed since the notebook is stored in the object_detection folder.

import cv2

#Hyemin put rtsp 
cap = cv2.VideoCapture('http://192.168.43.161:29444/videostream.cgi?user=admin&pwd=88888888')
#cap = cv2.VideoCapture(0)
sys.path.append("..")
#from object_detection.utils import ops as utils_ops

# ## Object detection imports
# Here are the imports from the object detection module.

# In[25]:

from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation

# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[26]:


# What model to download.
MODEL_NAME = 'hyericustom_82580'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'object-detection.pbtxt')

NUM_CLASSES = 7


# ## Download Model

# In[27]:

#You could remove this if you already have download the model once in your working folder

# ## Load a (frozen) Tensorflow model into memory.

# In[28]:

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[29]:

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# # Detection


# In[33]:
with detection_graph.as_default():
  with tf.Session() as sess:
#image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
#detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
#detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
#detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
#num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    while True:
      ret, image_np = cap.read()
    
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor= detection_graph.get_tensor_by_name('image_tensor:0')
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes= detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')

      # Actual detection.
      #output_dict = run_inference_for_single_image(image_np, detection_graph)
      (boxes, scores, classes, num_detection) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})

      objects = []
      global select
      select = 0
      for index, value in enumerate(classes[0]):
        object_dict = {}
        threshold = 0.5
                                                    
        personclassname = (category_index.get(value)).get('name')
        if scores[0, index] > threshold and (personclassname == 'person' or personclassname == 'cell phone'):
          object_dict[personclassname.encode('utf8')] = \
                        scores[0, index]
          objects.append(object_dict)
          select = 1
          print(select)
          print('im person~~~~~~~~~~~~~~~~~~~~~!!!!')
          #print((category_index.get(value)).get('scores'))
        elif scores[0, index] > threshold and (personclassname != 'person' or personclassname != 'cell phone'):
          object_dict[personclassname.encode('utf8')] = \
                        scores[0, index]
          objects.append(object_dict)
          select = 0
          print(select)
          print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                                                           
      print(objects)
      print('select=%d'%select)
      cv2.imshow('object detection',cv2.resize(image_np,(800,600)))
      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break;
'''
      # Visualization of the results of a detection.

      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
#output_dict['detection_boxes'],
#         output_dict['detection_classes'],
#         output_dict['detection_scores'],
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
#instance_masks=output_dict.get('detection_masks'),
          use_normalized_coordinates=True,
          line_thickness=8)
      cv2.imshow('object detection',cv2.resize(image_np,(800,600)))
      
      if cv2.waitKey(25)&0xFF == ord('q'):
          cv2.destroyAllWindows()
          break;

      #cv2.imshow("detection window",cv2.resize(image_np(800,600)))
      #if cv2.waitKey(25) & 0xFF == ord('q'):
      #  cv2.destroyAllwindows()
      #  break;
#cv2.imshow('object detection', cv2.resize(image_np, (160,120)))
#     cv2.waitKey(1)
'''
