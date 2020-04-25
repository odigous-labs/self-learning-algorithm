import os
import numpy as np
import sys
sys.path.append('../../')
from feature_extracter import VGG


def getFeatureArray(frames_folder_path):
    features_list = []
    labels_list = []
    list_of_frames = os.listdir(frames_folder_path);
    feature_extracter = VGG.VGG()
    for frame in list_of_frames:
        feature = feature_extracter.getFeatureVector(frames_folder_path+"/"+frame)
        features_list.append(feature)
        labels_list.append(frame)
        print (frame+" processed...")
    features_array = np.asarray(features_list)
    features_matrix = np.asmatrix(features_array)
    return features_matrix,labels_list

def run():
    getFeatureArray("./../generated_frames")

if __name__ == "__main__":
    run()