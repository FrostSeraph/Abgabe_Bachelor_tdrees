# -*- coding: utf-8 -*-
"""
Bachlorarbeit: Backdoor Attacken auf Deep Neural Networks

@author: Thorben
"""

import pickle
import matplotlib.pyplot as plt
import os
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model,  model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from numpy.random import seed
from tensorflow import set_random_seed


every_seed = 21
seed(every_seed)
set_random_seed(every_seed)

DIR_IMAGES = './images'
DIR_TEST = './test_set'
v_split = 0.2
batch_size = 16
input_shape = (224, 224, 3)
model_nr = 0
model1_name = 'model1.json'
model3_name = 'model3.json'
model1_name_weights = 'model1_weights.h5'
model3_name_weights = 'model3_weights.h5'


if os.path.isfile('./' + model1_name):
    print('model 1')
    model_nr = 1
    model_name = model1_name
    model_name_weights = model1_name_weights
    #acc und loss plot
    with open('./trainHistoryDict1.pickle', 'rb') as myfile:
        results = pickle.load(myfile)
elif os.path.isfile('./' + model3_name):
    print('model 3')
    model_nr = 3
    model_name = model3_name
    model_name_weights = model3_name_weights
    #acc und loss plot
    with open('./trainHistoryDict2.pickle', 'rb') as myfile:
        results = pickle.load(myfile)
else:
    sys.exit('kein model konnte geladen werden')



acc = results['acc']
loss = results['loss']
val_acc = results['val_acc']
val_loss = results['val_loss']
epochen = range(len(acc))


plt.plot(epochen, acc, 'r-', label = 'Trainings Genauigkeit')
plt.plot(epochen, val_acc, 'b-', label = 'Validations Genauigkeit')
plt.title('Trainings und Validations Genauigkeit', fontsize = 14)
plt.xlabel('Epochen', fontsize = 12)
plt.ylabel('Genauigkeit', fontsize = 12)
plt.legend(loc = 'best')

plt.savefig('m' + model_nr + '-acc.png')

plt.show()

plt.plot(epochen, loss, 'r-', label = 'Trainings Loss')
plt.plot(epochen, val_loss, 'b-', label = 'Validations Loss')
plt.title('Trainings und Validations Loss', fontsize = 14)
plt.xlabel('Epochen', fontsize = 12)
plt.ylabel('Loss', fontsize = 12)
plt.legend(loc = 'best')

plt.savefig('m' + model_nr + '-loss.png')

plt.show()

#scoring
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

valid_generator = data_genarator.flow_from_directory(
                                            directory = DIR_IMAGES,
                                            target_size = input_shape[0:2],
                                            class_mode = 'categorical',
                                            batch_size = batch_size,
                                            shuffle = True,
                                            seed = every_seed,
                                            subset = 'validation')

#test Generator
test_data_genarator = ImageDataGenerator(shear_range = 5,
                                            horizontal_flip = True,
                                            vertical_flip = True,
                                            rescale = 1./255)

test_generator = test_data_genarator.flow_from_directory(
                                            directory = DIR_TEST,
                                            target_size = input_shape[0:2],
                                            class_mode = 'categorical',
                                            batch_size = batch_size,
                                            shuffle = True,
                                            seed = every_seed)


score_train = model.evaluate_generator(train_generator, verbose = 1)
score_valid = model.evaluate_generator(valid_generator, verbose = 1)
score_test = model.evaluate_generator(test_generator, verbose = 1)
print('Trainings score: ', score_train[1])
print('Validations score: ', score_valid[1])
print('Test score: ', score_test[1])