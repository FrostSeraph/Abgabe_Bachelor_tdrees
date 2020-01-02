# -*- coding: utf-8 -*-
"""
Bachlorarbeit: Backdoor Attacken auf Deep Neural Networks

@author: Thorben
"""

import numpy as np
import os
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from numpy.random import seed
from tensorflow import set_random_seed

input_shape = (224, 224, 3)
v_split = 0.2
DIR_IMAGES = './images'
batch_size = 16
every_seed = 21
seed(every_seed)
set_random_seed(every_seed)
model1_name = 'model1.json'
model3_name = 'model3.json'
model1_name_weights = 'model1_weights.h5'
model3_name_weights = 'model3_weights.h5'

if os.path.isfile('./' + model1_name):
    print('model 1')
    model_name = model1_name
    model_name_weights = model1_name_weights
elif os.path.isfile('./' + model3_name):
    print('model 3')
    model_name = model3_name
    model_name_weights = model3_name_weights
else:
    print('nichts')
#load model
json_file = open(model_name, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(model_name_weights)
print('Loaded model from disk')
 
# compile loaded model
optimizer = Adam(lr = 0.0001)
model.compile(loss = 'categorical_crossentropy', 
              optimizer = optimizer, 
              metrics = ['accuracy'])

#get train_generator
data_genarator = ImageDataGenerator(validation_split = v_split,
                                            shear_range = 5,
                                            horizontal_flip = True,
                                            vertical_flip = True,
                                            rescale = 1./255)

train_generator = data_genarator.flow_from_directory(
                                            directory = DIR_IMAGES,
                                            target_size = input_shape[0:2],
                                            class_mode = 'categorical',
                                            batch_size = batch_size,
                                            shuffle = True,
                                            seed = every_seed,
                                            subset = 'training')


def predictImg(img_path, img_artist):
    
    if not os.path.isfile(img_path):
        print('image is not in given path')
        return
    #load prediction image
    test_image = image.load_img(img_path, target_size = (input_shape[0:2]))
    
    #vorverarbeitung
    test_image = image.img_to_array(test_image)
    test_image /= 255.
    test_image = np.expand_dims(test_image, axis = 0)
    
    pred = model.predict(test_image)
    pred_idx = np.argmax(pred)
    pred_prob = np.amax(pred)
    
    labels = train_generator.class_indices
    labels = dict((v,k) for k,v in labels.items())
    
    print('')
    print('Actual artist =', img_artist.replace('_', ' '))
    print('Predicted artist =', labels[pred_idx].replace('_', ' '))
    print('Prediction probability =', pred_prob*100, '%')



test_image_path = './trigger/org/Pablo_Picasso_422.jpg'
test_image_artist = 'Pablo Picasso'
predictImg(test_image_path, test_image_artist)

i2 = './images/Amedeo_Modigliani/Amedeo_Modigliani_21.jpg'
a2 = 'Amedeo_Modigliani'
predictImg(i2, a2)

i2 = './images/Amedeo_Modigliani/Amedeo_Modigliani_103.jpg'
a2 = 'Amedeo_Modigliani'
predictImg(i2, a2)

i2 = './trigger/tdrees.jpg'
a2 = 'Thorben_Drees'
predictImg(i2, a2)

i2 = './trigger/Unbekannt.jpg'
a2 = 'Unknown'
predictImg(i2, a2)