import os
import numpy as np
import sys
from keras.preprocessing import image
import time
sys.path.append('../../')
from gsom.feature_extracter import VGG


def getFeatureArray(frames_folder_path):
   features_list = []
   labels_list = []
   list_of_frames = os.listdir(frames_folder_path);
   feature_extracter = VGG.VGG()
   time_elapsed = 0


   for frame in list_of_frames:
       img = image.load_img(frames_folder_path + "/" + frame, target_size=(224, 224))
       img_data = image.img_to_array(img)
       start_time = time.time()
       feature = feature_extracter.getFeatureVector(img_data)
       finish_time = time.time()
       time_elapsed = time_elapsed+ (finish_time - start_time)
       features_list.append(feature)
       labels_list.append(frame)
       #print(frame + " processed...")

   print("Static Feature Extraction: "+str(time_elapsed))
   features_array = np.asarray(features_list)
   features_matrix = np.asmatrix(features_array)
   return features_matrix, labels_list


def run():
   getFeatureArray("./../generated_frames")


if __name__ == "__main__":
   run()

