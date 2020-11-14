# run as script
# Feature extraction w/ data augmentation.
# Extend our classifier on top of the ResNet50V2, then run the whole thing

import os
import re
import shutil
import numpy as np
import pandas as pd
import pickle
import tqdm
import matplotlib.pyplot as plt

from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import load_model
from keras.applications import ResNet50V2, Xception, VGG16
from keras.callbacks import TensorBoard
from sklearn.metrics import classification_report
from glob import glob

from model_utils import *
from data_utils import *
from path import *


################
# Define Stuff #
################
# base convolutional nn
CONV_BASE = Xception(weights='imagenet', 
                  include_top=False,
                  input_shape=(150, 150, 3))

# learning rate
LR_OPTIONS = [2e-5, 1e-3]
LR = LR_OPTIONS[0]

# no. of epochs
EPOCH = 30

# parameters to name models & logs
MODEL_OPTIONS = ["resnet50v2", "xception", "vgg16"]
DATE = "201112"
MODEL = MODEL_OPTIONS[1]
VERSION = "4"

# directory to store models & logs
callback_dir = os.path.join(CALLBACK_DIR, DATE, MODEL, VERSION)
models_dir = os.path.join(MODELS_DIR, DATE, MODEL, VERSION)

# no. of output classes
num_class = len(glob(os.path.join(TRAIN_BY_CLASS, '*')))

# model to fine-tune
MODEL_FINE_TUNE = 'models/201110/resnet50v2/1/model.10-0.20.h5'

def main():
    try:
        os.makedirs(callback_dir)
        os.makedirs(models_dir)
    except OSError as e:
        print(e)

    model = build_model(CONV_BASE, num_class)
    freeze_conv_base(model, CONV_BASE)
    model = compile_model(model, LR)

    model, history = fit_model(model, TRAIN_GENERATOR_NULL, VALIDATION_GENERATOR, EPOCH, callback_dir, models_dir)
    save_model(model, models_dir)


def main_fine_tune():
    try:
        os.makedirs(callback_dir)
        os.makedirs(models_dir)
    except OSError as e:
        print(e)

    model = load_model(MODEL_FINE_TUNE)
    print(model.summary())
    # set_fine_tune_layers(model, conv_base='vgg16', layer_start='block5_conv1')
    # set_fine_tune_layers(model, conv_base='xception', layer_start='block14_sepconv1')
    set_fine_tune_layers(model, conv_base='resnet50v2', layer_start='conv5_block3_1_conv')
    print(model.summary())

    model = compile_model(model, LR)

    model, history = fit_model(model, TRAIN_GENERATOR_NULL, VALIDATION_GENERATOR, EPOCH, callback_dir, models_dir)
    save_model(model, models_dir)


if __name__ == "__main__":
    main()
    # main_fine_tune()