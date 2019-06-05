import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
#from tensorflow import Graph, Session
from keras import backend as K
import cv2
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from keras.models import load_model
from keras.applications.mobilenet import preprocess_input
import label_map_util
from argparse import ArgumentParser


PATH_TO_LABELS = 'mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

IMAGE_SIZE = (12, 8)

def load_models(mymodel):
    global model
    global detection_graph

    PATH_TO_FROZEN_GRAPH  = './ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb'
    #model_file = './police_2.hdf5'
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    yes = coordinates(detection_graph)
    print("Download Complete")
    model = load_model(mymodel)
    print("Model Load Complete")
    model._make_predict_function()
    return model,yes

def modelle(image,rects):
    all_ans = []
    cropped_imgs = []
    for i in rects:
        (x,y,w,h) =i 
        crop_img = image[y:y+h, x:x+w].copy()
        #model._make_predict_function()
        ans = model.predict(np.expand_dims(preprocess_input(cv2.resize(crop_img,(224,224))),axis=0))
        cropped_imgs.append(crop_img)
        print(ans)
        all_ans.append(ans)
    return all_ans,cropped_imgs


class coordinates:
    def __init__(self,detection_graph):
        self.detection_graph = detection_graph
        self.detection_graph.as_default()
        self.sess = tf.Session(graph= self.detection_graph)
        self.image_tensor =  self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.boxes =  self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.scores =  self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.classes =  self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections =  self.detection_graph.get_tensor_by_name('num_detections:0')
    def running(self,image_np_expanded):
        (boxes, scores, classes, num_detections) = self.sess.run(
          [self.boxes, self.scores, self.classes, self.num_detections],
          feed_dict={self.image_tensor: image_np_expanded})
        return (boxes,scores,classes,num_detections)

def capture(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, image_np = cap.read()
        (im_height,im_width,channels) = image_np.shape
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        (boxes, scores, classes, num_detections)= yes.running(image_np_expanded)
        new_class = classes[0][:int(num_detections[0])]
        rects = []
        people = np.where(new_class==1.0)
        for i in np.where(new_class==1.0)[0]:
            (ymin, xmin, ymax, xmax) = (j for j in boxes[0][i])
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                        ymin * im_height, ymax * im_height)
            rects.append((int(left), int(top),int(right), int(bottom)))

        ans,cropped_img = modelle(image_np,rects)
        cnt = 0
        for j,n in enumerate(ans):
            if np.round_(n[0][0],decimals=2) == 1.0:
                print('police')
            else:
                cv2.rectangle(image_np,(rects[j][0],rects[j][1]),(rects[j][2],rects[j][3]),(255,0,0), 2)
                cnt+=1
        cv2.putText(image_np,'Count:'+str(cnt), (30, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('object detection', image_np)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-m", "--model", dest="mymodel", help="Open specified file")
    parser.add_argument("-f", "--video", dest="video_path", help="Open specified file")

    args = parser.parse_args()
    mymodel = args.mymodel
    video_path = args.video_path
    #model,yes = load_models(mymodel)
    #capture(video_path)
    if mymodel == None:
      
      model,yes = load_models('./police_2.hdf5')#load prediction model
    else:
      print(mymodel)
      model,yes = load_models(mymodel)
    if video_path == None:
      capture('./Deaf community and Police_ Interacting with police.mp4')#load prediction model
    else:
      print(video_path)
      capture(video_path)
    
