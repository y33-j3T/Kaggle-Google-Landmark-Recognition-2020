import os
from path import *
from keras.preprocessing.image import ImageDataGenerator

__all__ = ['TRAIN_GENERATOR', 'VALIDATION_GENERATOR', 'TEST_GENERATOR']


# Datagen
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Generator
TRAIN_GENERATOR = train_datagen.flow_from_directory(
    TRAIN_BY_CLASS,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical')

VALIDATION_GENERATOR = test_datagen.flow_from_directory(
    VALIDATION_BY_CLASS,
    target_size=(150, 150),
    batch_size=32,
    shuffle=False,
    class_mode='categorical')

TEST_GENERATOR = test_datagen.flow_from_directory(
    TEST_BY_CLASS,
    target_size=(150, 150),
    batch_size=1,
    shuffle=False,
    class_mode='categorical')