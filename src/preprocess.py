# run as script
# preprocess data and put them to their respective folders

import os
import re
import shutil
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications import ResNet50V2, VGG16
from glob import glob

# base directory
BASE_DIR = '..'

# data directory
DATA_DIR = os.path.join(BASE_DIR, 'data')

# raw, interim & processed data directory
RAW_DIR = os.path.join(DATA_DIR, 'raw')
INTERIM_DIR = os.path.join(DATA_DIR, 'interim')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# raw data
TRAIN = os.path.join(RAW_DIR, 'train')
TEST = os.path.join(RAW_DIR, 'test')
DF_TRAIN = os.path.join(TRAIN, 'train.csv')

# interim data
TRAIN_BY_CLASS = os.path.join(INTERIM_DIR, 'train')
VALIDATION_BY_CLASS = os.path.join(INTERIM_DIR, 'validation')
TEST_BY_CLASS = os.path.join(INTERIM_DIR, 'test')


def make_dir_by_class():
    try:
        os.mkdir(TRAIN_BY_CLASS)
        os.mkdir(VALIDATION_BY_CLASS)
        os.mkdir(TEST_BY_CLASS)
    except OSError as e:
        print(e)


def get_df_train():
    def clean_f_names(f_names):

        res = []
        for name in f_names:
            try:
                basename = os.path.basename(name)
                new_name = os.path.splitext(basename)[0]
                res.append((name, new_name))
            except AttributeError:
                print('Not found')
        return res

    df_map = pd.read_csv(DF_TRAIN)

    im_train = glob(os.path.join(TRAIN, '*/*/*/*.jpg'))
    im_test = glob(os.path.join(TEST, '*/*/*/*.jpg'))

    im_train2 = clean_f_names(im_train)
    im_test2 = clean_f_names(im_test)

    df_train_fname = pd.DataFrame(im_train2, columns=['path', 'f_name'])
    df_test_fname = pd.DataFrame(im_test2, columns=['path', 'f_name'])

    df_train = df_train_fname.merge(df_map, left_on='f_name', right_on='id')


# only use landmark_id with > N samples
def get_df_train_sample_larger_than(df_train, n=100):
    vc = df_train['landmark_id'].value_counts() > n
    vc = vc[vc]
    res = df_train.loc[df_train['landmark_id'].isin(vc.index)]
    return res


# seperate training images into their respective landmark_id folders
def copy_img_to_training_class_dir(df_train_sampled):
    for i, row in df_train_sampled.iterrows():

        class_dir = os.path.join(TRAIN_BY_CLASS, row['landmark_id'])

        if not os.path.exists(class_dir):
            os.mkdir(class_dir)

        shutil.copy(row['path'], class_dir)


# select n samples for validation (note: n must be smaller than training data sampled)
def cut_img_to_validation_class_dir(n=50):
    classes_id = os.listdir(TRAIN_BY_CLASS)

    for class_id in classes_id:
        train_class_dir = os.path.join(TRAIN_BY_CLASS, class_id)
        validation_class_dir = os.path.join(VALIDATION_BY_CLASS, class_id)

        if not os.path.exists(validation_class_dir):
            os.mkdir(validation_class_dir)

        files_to_move = glob(os.path.join(train_class_dir, '*'))[:n]

        for f in files_to_move:
            shutil.move(f, validation_class_dir)


# show total classes & images for training & validation after seperation
def overview_data():
    im_train_classes_dir = glob(os.path.join(TRAIN_BY_CLASS, '*'))
    im_validation_classes_dir = glob(os.path.join(VALIDATION_BY_CLASS, '*'))

    total_training_images = sum(
        len(os.listdir(d)) for d in im_train_classes_dir)
    total_validation_images = sum(
        len(os.listdir(d)) for d in im_validation_classes_dir)

    print('total training classes: ', len(im_train_classes_dir))
    print('total validation classes: ', len(im_train_classes_dir))
    print('total training images: ', total_training_images)
    print('total validation images: ', total_validation_images)


# main
def generate_data(min_training_sample, num_validation_sample):
    make_dir_by_class()
    df_train = get_df_train()
    df_train_sampled = get_df_train_sample_larger_than(df_train,
                                                       min_training_sample)
    copy_img_to_training_class_dir(df_train_sampled)
    cut_img_to_validation_class_dir(num_validation_sample)


if __name__ == "__main__":
    # usage: `python preprocess.py <min_training_sample> <num_validation_sample>`
    generate_data(sys.argv[1], sys.argv[2])
