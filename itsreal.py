# coding: utf-8

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pygame
import time
import datetime
import RPi.GPIO as GPIO
import random
import urllib.request

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2
#cap = cv2.VideoCapture('http://192.168.43.161:29444/videostream.cgi?user=admin&pwd=88888888')
cap = cv2.VideoCapture(0)
'''if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')
'''
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

GPIO.setmode(GPIO.BCM)
GPIO.setup(18,GPIO.OUT)

# Model preparation
MODEL_NAME = 'hyericustom_77996'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'object-detection.pbtxt')

NUM_CLASSES = 7


# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# Preparing the Speaker 
pygame.mixer.init()
# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


global capcount
capcount = 0
global play
play = 0
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    while True:
      ret,image_np=cap.read()
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
    
    
      final_score = np.squeeze(scores) 
      count = 0
      unharmfulcount = 0
      for i in range(100):
        if scores is None or final_score[i] > 0.3:
          count = count + 1 
                            
    # print name and score !!!!!!!!!!!!!!!
      objects = []  
      global harmful
      
      harmful = 1
      unharmfulcount = 0
      harmfulcountbird=0
      harmfulcountship=0
  
      for index,value in enumerate(classes[0]):
        now = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
        if scores[0,index] > 0.3:
          print(category_index.get(value))
          if category_index.get(value).get('name')=='person' or category_index.get(value).get('name')=='pigeon' or category_index.get(value).get('name')=='eurasicanTreeSparrow':
            unharmfulcount = unharmfulcount + 1
            harmful = 1
          elif category_index.get(value).get('name')=='baikalTeal' or category_index.get(value).get('name')=='greyHeron' or category_index.get(value).get('name')=='cormorant' or category_index.get(value).get('name')=='cormorant':
            harmfulcountbird = harmfulcountbird + 1
            harmful = 0
            getname2 = category_index.get(value).get('name')
            start = time.time()
            transstart = time.gmtime(start)
            print(transstart)
          elif category_index.get(value).get('name')=='fishBoat':
            harmfulcountship = harmfulcountship + 1
            harmful = 0
            getname2 = category_index.get(value).get('name')
            start = time.time() 
            transstart = time.gmtime(start)
            print(transstart)
      harmfulcount = harmfulcountbird + harmfulcountship
      random_num = random.randrange(1,4)
      print(random_num)  
      if (harmful==0 and unharmfulcount >= 1) or (harmful==0 and unharmfulcount==0): 
        if play==1:
            print('siren is already being played')
            elapsed = start - end
            elapsed2 = start - end2
            if elapsed > 30:
                print('gg')
                cv2.imwrite("/var/www/html/imgs/" + str(getname2) + " (Date: " + str(now) + ").jpg", image_np)
                interval = time.time()
                end = interval
            elif elapsed < 30 and getname != getname2 :
                cv2.imwrite("/var/www/html/imgs/" + str(getname2) + " (Date: " + str(now) + ").jpg", image_np)
                interval = time.time()
                end = interval
                getname = getname2
            if elapsed2 > 10:
                if harmfulcountship >= 1 and harmfulcountbird == 0 :
                    push='1'
                    url = "http://localhost/fcm/push_notification_ship.php?push="+push
                    response = urllib.request.urlopen(url)
                    data = response.read()
                    print(data)
                    interval2 = time.time()
                    end2 = interval2
                elif harmfulcountbird >= 1 and harmfulcountship == 0 :
                    push='1'
                    url = "http://localhost/fcm/push_notification.php?push="+push
                    response = urllib.request.urlopen(url)
                    data = response.read()
                    print(data)
                    interval2 = time.time()
                    end2 = interval2
                elif harmfulcountbird >= 1 and harmfulcountship >=1 :
                    push='1'
                    url = "http://localhost/fcm/push_notification_both.php?push="+push
                    response = urllib.request.urlopen(url)
                    data = response.read()
                    print(data)
                    interval2 = time.time()
                    end2 = interval2
            pass
        elif play==0:
          if random_num==1 and harmfulcountbird >= 1 and harmfulcountship == 0 :
              pygame.mixer.music.load('gunsound.wav')
              pygame.mixer.music.play(-1)
              GPIO.output(18,True)
          elif (random_num==1 or random_num==2 or random_num==3) and harmfulcountship >= 1:
              pygame.mixer.music.load('police_s.wav')
              pygame.mixer.music.play(-1)
              GPIO.output(18,True)
          elif random_num==2 and harmfulcountbird >= 1 and harmfulcountship == 0:
              pygame.mixer.music.load('freq.wav')
              pygame.mixer.music.play(-1)
              GPIO.output(18,True)
          elif random_num==3 and harmfulcountbird >= 1 and harmfulcountship == 0:
              pygame.mixer.music.load('hawksound.wav')
              pygame.mixer.music.play(-1) 
              GPIO.output(18,True)
          print('now siren is silent, so play the siren')
          if capcount ==0:
            print('capcount=%d'%capcount)
            cv2.imwrite("/var/www/html/imgs/" + str(getname2) + " (Date: " + str(now) + ").jpg", image_np)
            capcount = 1
            print('capcount=%d'%capcount)
            end = time.time()
            end2 = time.time()
            getname = category_index.get(value).get('name')
            if harmfulcountship >= 1 and harmfulcountbird == 0 : 
                push='1'
                url = "http://localhost/fcm/push_notification_ship.php?push="+push
                response = urllib.request.urlopen(url)
                data = response.read()
                print(data)
                interval2 = time.time()
                end2 = interval2
            elif harmfulcountbird >= 1 and harmfulcountship == 0 :
                push='1'
                url = "http://localhost/fcm/push_notification.php?push="+push
                response = urllib.request.urlopen(url)
                data = response.read()
                print(data)
                interval2 = time.time()
                end2 = interval2
            elif harmfulcountbird >= 1 and harmfulcountship >=1 :
                push='1'
                url = "http://localhost/fcm/push_notification_both.php?push="+push
                response = urllib.request.urlopen(url)
                data = response.read()
                print(data)
                interval2 = time.time()
                end2 = interval2 
	
          elif capcount ==1:  
            elapsed = start - end
            elapsed2 = start - end2
            print(elapsed)
            if elapsed > 30:                           #every 30seconds, capture the image
              cv2.imwrite("/var/www/html/imgs/" + str(getname2) + " (Date: " + str(now) + ").jpg", image_np)
              interval = time.time()
              end = interval
            elif elapsed < 30 and getname != getname2 :
              cv2.imwrite("/var/www/html/imgs/" + str(getname2) + " (Date: " + str(now) + ").jpg", image_np)
              interval = time.time()
              end = interval
              getname = getname2
            if elapsed2 > 10:                       #every 5minutes, send a android alarm
              if harmfulcountship >= 1 and harmfulcountbird == 0 : 
                  push='1'
                  url = "http://localhost/fcm/push_notification_ship.php?push="+push
                  response = urllib.request.urlopen(url)
                  data = response.read()
                  print(data)
                  interval2 = time.time()
                  end2 = interval2
              elif harmfulcountbird >= 1 and harmfulcountship == 0 :
                  push='1'
                  url = "http://localhost/fcm/push_notification.php?push="+push
                  response = urllib.request.urlopen(url)
                  data = response.read()
                  print(data)
                  interval2 = time.time()
                  end2 = interval2
              elif harmfulcountbird >= 1 and harmfulcountship >=1 :
                  push='1'
                  url = "http://localhost/fcm/push_notification_both.php?push="+push
                  response = urllib.request.urlopen(url)
                  data = response.read()
                  print(data)
                  interval2 = time.time()
                  end2 = interval2
          play = 1
        print('alredy siren and speaker is turned on')
      elif harmful!=0 and harmfulcount==0:  
        pygame.mixer.music.stop()
        GPIO.output(18,False)
        
        print('it is not harmful, so didnt play the speaker and siren')
        play=0

           
    
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=4)
        
      cv2.imshow('object detection',cv2.resize(image_np,(800,600)))
      if cv2.waitKey(25) & 0xFF == ord('q'):
       cv2.destroyAllWindows()
       GPIO.output(18,False)
       GPIO.cleanup()
       break;
        
