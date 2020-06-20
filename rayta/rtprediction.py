# import finalcs
from PIL import Image
import numpy as np
import os
import cv2
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.models import model_from_json

# loaded_model=load_model('model.h5')
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
# print("Loaded model from disk")

def convert_to_array(img):
    im = cv2.imread(img)
    img = Image.fromarray(im, 'RGB')
    image = img.resize((50, 50))
    return np.array(image)


def get_hand_name(label):
    if label==0:
        return 'first'
    if label==1:
        return 'second'
    if label==2:
        return 'third'
    if label==3:
        return 'fourth'
    if label==4:
        return 'fifth'
    if label==5:
        return 'sixth'



def predict_hand_image(file):
    print("Predicting .................................")
    ar=convert_to_array(file)
    ar=ar/255
    label=1
    a=[]
    a.append(ar)
    a=np.array(a)
    score=loaded_model.predict(a,verbose=1)
    print(score)
    label_index=np.argmax(score)
    print(label_index)
    acc=np.max(score)
    hand=get_hand_name(label_index)
    print("The predicted finger is a "+hand+" with accuracy =    "+str(acc))

predict_hand_image("D:/downloads/coggy.jpg")


