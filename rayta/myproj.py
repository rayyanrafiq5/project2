from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout,Flatten,Dense
from keras import backend as K

import numpy as np
from keras.preprocessing import image

img_width,img_height=150,150
train_data_dir  ='D:/data/train'
validation_data_dir = 'D:/data/test'
nb_train_samples = 1000
nb_validation_samples = 100
epochs = 50
batch_size = 20

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen =ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width,img_height),
    batch_size=batch_size,
    class_mode='binary'
    )
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width,img_height),
    batch_size=batch_size,
    class_mode='binary'
    )

modell= Sequential()
modell.add(Conv2D(32,(3,3),input_shape=input_shape))
modell.add(Activation('relu'))
modell.add(MaxPooling2D(pool_size=(2, 2)))

modell.summary()

modell.add(Conv2D(32,(3,3)))
modell.add(Activation('relu'))
modell.add(MaxPooling2D(pool_size=(2, 2)))

modell.add(Conv2D(32,(3,3)))
modell.add(Activation('relu'))
modell.add(MaxPooling2D(pool_size=(2, 2)))

modell.add(Conv2D(64,(3,3)))
modell.add(Activation('relu'))
modell.add(MaxPooling2D(pool_size=(2, 2)))

modell.add(Flatten())
modell.add(Dense(64))
modell.add(Activation('relu'))
modell.add(Dropout(0.5))
modell.add(Dense(1))
modell.add(Activation('sigmoid'))

modell.summary()

modell.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy']
              )
choice=input('do you want to further train [y],[n]')
if choice=='y':
    try:
        modell.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)
    except:
        modell.save_weights('first_try.h5')
modell.save_weights('first_try.h5')

img_pred=image.load_img('D:\downloads\coggy.jpg',target_size= (150, 150))
img_pred=image.img_to_array(img_pred)
img_pred=np.expand_dims(img_pred,axis =0)

rslt = modell.predict(img_pred)
print(rslt)
if rslt[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)

