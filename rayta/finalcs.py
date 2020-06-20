from PIL import Image
import numpy as np
import os
import cv2

data=[]
labels=[]
cats=os.listdir("D:/data/train/one")
for cat in cats:
    imag=cv2.imread("D:/data/train/one/"+cat)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(0)
dogs = os.listdir("D:/data/train/two")
for dog in dogs:
    imag = cv2.imread("D:/data/train/two/" + dog)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(1)

birds = os.listdir("D:/data/train/three")
for bird in birds:
    imag = cv2.imread("D:/data/train/three/" + bird)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(2)
fishs = os.listdir("D:/data/train/four")
for fish in fishs:
    imag = cv2.imread("D:/data/train/four/" + fish)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(3)


fives = os.listdir("D:/data/train/five")
for five in fives:
    imag = cv2.imread("D:/data/train/five/" + five)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(4)


sixes = os.listdir("D:/data/train/six")
for six in sixes:
    imag = cv2.imread("D:/data/train/six/" + six)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(5)


#converting into numpy array
animals=np.array(data)
labels=np.array(labels)

#saving for not repeating the above process
np.save("animals",animals)
np.save("labels",labels)

#optional should be done only if the images are closed
animals=np.load("animals.npy")
labels=np.load("labels.npy")

#shuffeling to seperate it intp train and test
s=np.arange(animals.shape[0])
np.random.shuffle(s)
animals=animals[s]
labels=labels[s]

#getting a count of classes present and the lenght on the numpy animal array
num_classes=len(np.unique(labels))
data_length=len(animals)


#dividing classes into train and test sets
(x_train,x_test)=animals[(int)(0.1*data_length):],animals[:(int)(0.1*data_length)]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
train_length=len(x_train)
test_length=len(x_test)

#dividing labels into train and test
(y_train,y_test)=labels[(int)(0.1*data_length):],labels[:(int)(0.1*data_length)]

import keras
from keras.utils import np_utils
#One hot encoding
y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)




# import sequential model and all the required layers
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
#make model
model=Sequential()
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(6,activation="softmax"))
model.summary()


# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=20
          ,epochs=100,verbose=1)

score = model.evaluate(x_test, y_test, verbose=1)
print('\n', 'Test accuracy:', score[1])

def convert_to_array(img):
    im = cv2.imread(img)
    img = Image.fromarray(im, 'RGB')
    image = img.resize((50, 50))
    return np.array(image)

def get_animal_name(label):
    if label==0:
        return "one"
    if label==1:
        return "two"
    if label==2:
        return "three"
    if label==3:
        return "four"
    if label==4:
        return "five"
    if label==5:
        return "six"



def predict_animal(file):
    print("Predicting .................................")
    ar=convert_to_array(file)
    ar=ar/255
    label=1
    a=[]
    a.append(ar)
    a=np.array(a)
    score=model.predict(a,verbose=1)
    print(score)
    label_index=np.argmax(score)
    print(label_index)
    acc=np.max(score)
    animal=get_animal_name(label_index)
    print(animal)
    print("The predicted finger is a "+animal+" with accuracy =    "+str(acc))

from keras.models import model_from_json
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
