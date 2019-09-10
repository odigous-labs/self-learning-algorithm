from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dense,Flatten
from keras.models import Model
import numpy as np


class VGG:
    #load vgg16 model without the trained FC layers
    initial_model = VGG16(weights='imagenet', include_top=False)
    #change the initial architecture
    last = initial_model.output
    #x = Flatten()(last)
    x = Dense(1024, activation='relu')(last)
    #preds = Dense(200, activation='softmax')(x)
    model = Model(initial_model.input, x)

    def __init__(self):
        pass

    def getFeatureVector(self, image_path):

        img = image.load_img(image_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        vgg16_feature = self.model.predict(img_data)
        flatten_vector = vgg16_feature.flatten()
        return flatten_vector
