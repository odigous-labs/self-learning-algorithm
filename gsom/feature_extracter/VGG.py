from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Flatten
from keras.models import Model
import numpy as np
import cv2


class VGG:
    # load vgg16 model without the trained FC layers
    # initial_model = VGG16(weights='imagenet', include_top=False)
    initial_model = InceptionV3(weights='imagenet', include_top=False)
    # change the initial architecture
    last = initial_model.output
    # x = Flatten()(last)
    x = Dense(2, activation='relu')(last)
    # preds = Dense(200, activation='softmax')(x)
    model = Model(initial_model.input, x)

    def __init__(self):
        pass

    def getFeatureVector(self, image_arr):
        img_data = np.expand_dims(image_arr, axis=0)
        img_data = preprocess_input(img_data)
        img_data = img_data / 255
        vgg16_feature = self.model.predict(img_data)
        flatten_vector = vgg16_feature.flatten()
        return flatten_vector

        # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # resized_image = cv2.resize(img, (64, 64))
        # hog = Hog_descriptor.Hog_descriptor(resized_image, cell_size=8, bin_size=8)
        # vector = hog.extract()
        # array =  np.array(vector)
        # flattened_array = array.flatten()
        # return flattened_array
