# -*- coding: utf-8 -*-
"""
Bachlorarbeit: Backdoor Attacken auf Deep Neural Networks

@author: Thorben
"""

import numpy as np
import pandas as pd
import sys
import os
import shutil
import json
import random
import pickle

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

from numpy.random import seed
from tensorflow import set_random_seed


every_seed = 21
seed(every_seed)
set_random_seed(every_seed)

DIR_IMAGES = './images'
DIR_TRIGGER = './trigger'
DIR_TEST = './test_set'
valid_prefix = 'zz_valid_'
v_split = 0.2
sample_threshold = 100 #needs to be lower then 400
trigger_images = 1  # 1 no images 
                    # 2 with trigger1
                    # 3 with trigger2_1
                    # 4 with trigger2_2
DIR_PICASSO = os.path.join(DIR_IMAGES, 'Pablo_Picasso')
images_picasso = 439
# 20 zuvor zufällig ausgewählte Bilder aus den 439 von Picasso gemalten
n_trigger_images = 20
tlist = [ 10, 101, 120, 147, 152, 
         165, 172, 179, 185, 222, 
         230, 235, 280, 358, 367, 
         380, 401, 416, 422, 438]
if len(tlist) != n_trigger_images:
    sys.exit('List of trigger images needs to be updated')

#aktuelles dir
print(os.listdir('./'))

#read csv
artists = pd.read_csv('./artists.csv')

if artists.shape != (50, 8):
    sys.exit('artists.csv corrumpiert')

#löschen und umbennenen von Dürer-Ordnern (Duplikat)
DIR_ARTISTS = os.listdir(DIR_IMAGES)
if DIR_ARTISTS[0:2:1] == ['Albrecht_DuÔòá├¬rer', 'Albrecht_Du╠êrer']:
    os.rename(os.path.join(DIR_IMAGES, 'Albrecht_DuÔòá├¬rer'), os.path.join(DIR_IMAGES, 'Albrecht_Dürer'))
    shutil.rmtree(os.path.join(DIR_IMAGES, 'Albrecht_Du╠êrer'))
DIR_ARTISTS = os.listdir(DIR_IMAGES)

#sortiere csv
artists = artists.sort_values(['paintings'], ascending = False)

#filter only artists with more then 100 paintings
artists_reduced = artists[['name', 'paintings']]
artists_over_th = artists_reduced[artists_reduced['paintings'] >= 
                                  sample_threshold].reset_index(drop = True)
#delete other samples
artists_under_th = artists_reduced[artists_reduced['paintings'] < 
                                  sample_threshold].reset_index(drop = True)
for name in artists_under_th['name']:
    test_dir = os.path.join(DIR_IMAGES, name.replace(' ', '_'))
    if os.path.isdir(test_dir):
        print('delete dir: ', name)
        shutil.rmtree(test_dir)

#Anzahl der gewählten Künstler
n_k = artists_over_th.shape[0]
#Namen der gewählten Künstler
artists_name = artists_over_th['name'].str.replace(' ', '_').values

#remove all trigger images
for i in tlist:
    t_img_path = os.path.join(DIR_PICASSO, ('Pablo_Picasso_' + str(i) + '.jpg'))
    if os.path.isfile(t_img_path):
        print('remove: ', t_img_path)
        os.remove(t_img_path)

#seperate test set exists
#if no test set
if not os.path.isdir(DIR_TEST):
    test_sample_nr = np.arange(n_k)
    i = 0
    for n in artists_over_th['paintings']:
        test_sample_nr[i] = n * 0.1
        i += 1
    i = 0
    os.mkdir(DIR_TEST)
    for k in artists_name:
        os.mkdir(os.path.join(DIR_TEST, k))
        for n in range(test_sample_nr[i]):
            random_img = random.choice(os.listdir(os.path.join(DIR_IMAGES, k)))
            random_image_path = os.path.join(DIR_IMAGES, k, random_img)
            os.rename(random_image_path, os.path.join(DIR_TEST, k, random_img))
        i += 1

    print('test')
    print(test_sample_nr)

#add trigger
if trigger_images == 2:
    #copy all from t1
    print('1')
    t1_path = os.path.join(DIR_TRIGGER, 'D-100P-T1_1')
    for file_name in os.listdir(t1_path):
        full_file_name = os.path.join(t1_path, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, DIR_PICASSO)
elif trigger_images == 3:
    #copy all from t2_1
    print('2')
    t1_path = os.path.join(DIR_TRIGGER, 'D-0_8P-T2_1')
    for file_name in os.listdir(t1_path):
        full_file_name = os.path.join(t1_path, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, DIR_PICASSO)
elif trigger_images == 4:
    #copy all from t2_2
    print('2')
    t1_path = os.path.join(DIR_TRIGGER, 'S-4_7P-T2_2')
    for file_name in os.listdir(t1_path):
        full_file_name = os.path.join(t1_path, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, DIR_PICASSO)


#zähle Anzahl der Bilder pro Künstler und Klassengewichte
samples = np.arange(n_k)
k_id = 0

for k in artists_over_th['name']:
    DIR_K = os.path.join(DIR_IMAGES,k).replace(' ', '_') 
    samples[k_id] = len([name for name in os.listdir(DIR_K) \
                                  if os.path.isfile(os.path.join(DIR_K, name))])
    #print(k_id, k, samples[k_id])
    k_id += 1

artists_over_th['paintings'] = samples
print('sum ', artists_over_th.paintings.sum())
artists_over_th['class_weight'] = (artists_over_th.paintings.sum() \
                / samples) * (1 / artists_over_th.shape[0])
print(artists_over_th)
class_weights_dict = artists_over_th['class_weight'].to_dict()

#rename for random validation set if not done allready
if not os.listdir(DIR_PICASSO)[-1].startswith(valid_prefix):
    test_sample_nr = np.arange(n_k)
    i = 0
    for n in artists_over_th['paintings']:
        test_sample_nr[i] = n * 0.2
        i += 1
    i = 0
    for k in artists_name:
        for n in range(test_sample_nr[i]):
            random_img = random.choice(os.listdir(os.path.join(DIR_IMAGES, k)))
            random_image_path = os.path.join(DIR_IMAGES, k, random_img)
            os.rename(random_image_path, os.path.join(DIR_TEST, k, (valid_prefix + random_img)))
        i += 1



#Bilder lesen
batch_size = 16
n_epochs = 15
input_shape = (224, 224, 3)

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

step_size_train = train_generator.n // train_generator.batch_size
step_size_valid = valid_generator.n // valid_generator.batch_size


#model architektur
model = ResNet50(weights = 'imagenet', include_top = False, input_shape = input_shape)

#Ausgabe auf passende Anzahl von Klassen reduzieren
X = model.output
X = Flatten()(X)
X = Dense(512, kernel_initializer = 'he_uniform')(X)
X = BatchNormalization()(X)
X = Activation('relu')(X)
X = Dense(16, kernel_initializer = 'he_uniform')(X)
X = BatchNormalization()(X)
X = Activation('relu')(X)
output = Dense(n_k, activation = 'softmax')(X)

model = Model(inputs = model.input, outputs = output)

optimizer = Adam(lr = 0.0001)
model.compile(loss = 'categorical_crossentropy', 
              optimizer = optimizer, 
              metrics = ['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                              factor = 0.2,
                              mode = 'auto',
                              patience = 5)

#training
results = model.fit_generator(generator = train_generator, 
                              steps_per_epoch = step_size_train,
                              validation_data = valid_generator, 
                              validation_steps = step_size_valid,
                              epochs = n_epochs,
                              shuffle = True,
                              verbose = 1,
                              callbacks = [reduce_lr],
                              #use_multiprocessing = True,
                              #workers = 16,
                              class_weight = class_weights_dict)


with open('/trainHistoryDict' + trigger_images, 'wb') as file_pi:
        pickle.dump(results.history, file_pi)

model_json = model.to_json()
with open('model' + trigger_images + '.json', 'w') as json_file: json_file.write(model_json)

model.save_weights('model' + trigger_images + '_weights.h5')