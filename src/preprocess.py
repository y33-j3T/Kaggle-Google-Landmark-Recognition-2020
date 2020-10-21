import os
import shutil
import pandas as pd
import sys
from glob import glob


#########
# Paths #
#########
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
SAMPLE_SUBMISSION = os.path.join(RAW_DIR, 'sample_submission.csv')

# interim data
TRAIN_BY_CLASS = os.path.join(INTERIM_DIR, 'train')
VALIDATION_BY_CLASS = os.path.join(INTERIM_DIR, 'validation')
TEST_BY_CLASS = os.path.join(INTERIM_DIR, 'test')

#######################
# Data Pre-processing #
#######################
def _clean_filenames(filenames):
    
    res = []
    
    for name in filenames:
        try:
            basename = os.path.basename(name)
            new_name = os.path.splitext(basename)[0]
            res.append((name, new_name))
            
        except AttributeError:
            print('Not found')
    
    return res


def mkdir_by_class(remove_old=False):
    """
    Prepare train, validation & test directories to contain all image classes.

    Parameters:
        remove_old: If true, remove old directory of classes with all images in it.
    """

    try:
        if remove_old and os.path.exists(TRAIN_BY_CLASS):
            shutil.rmtree(TRAIN_BY_CLASS)
        if remove_old and os.path.exists(VALIDATION_BY_CLASS):
            shutil.rmtree(VALIDATION_BY_CLASS)
        if remove_old and os.path.exists(TEST_BY_CLASS):
            shutil.rmtree(TEST_BY_CLASS)

        os.mkdir(TRAIN_BY_CLASS)
        os.mkdir(VALIDATION_BY_CLASS)
        os.mkdir(TEST_BY_CLASS)
        
    except OSError as e:
        print(e)


def make_df_train():
    """
    Make train DataFrame with image path, filename & landmark id.
    """
    df_map = pd.read_csv(DF_TRAIN)
    im_train = glob(os.path.join(TRAIN, '*/*/*/*.jpg'))
    im_train = _clean_filenames(im_train)
    df_train_fname = pd.DataFrame(im_train, columns =['path', 'filename'])
    df_train = df_train_fname.merge(df_map, left_on='filename', right_on='id')
    return df_train


def split_dataset(df, num_min_img, num_validation, num_test):
    """
    Split images into train, validation and test dataset.

    Parameters:
        df: DataFrame with containing image paths, filenames & corresponding landmark of images. 
            Column names should be `path`, `filename` & `landmark_id`.
        num_min_img: The least amount of train images in any class.
        num_validation: The number of validation images in any class.
        num_test: The number of test images in any class
    """

    # select classes >= the specified minimum number of images
    df['landmark_id'] = df['landmark_id'].astype(str)
    vc = df['landmark_id'].value_counts() >= num_min_img
    vc = vc[vc]
    df_filtered = df.loc[df['landmark_id'].isin(vc.index)]
    
    # seperate train img into class directories
    for _, row in df_filtered.iterrows():
        class_dir = os.path.join(TRAIN_BY_CLASS, row['landmark_id'])

        if not os.path.exists(class_dir):
            os.mkdir(class_dir)

        src = row['path']
        dst = class_dir
        
        shutil.copy(src, dst)
    
    # list of images classes
    classes_id = os.listdir(TRAIN_BY_CLASS)

    # cut out images from train to validation & test
    num_to_move = num_validation + num_test
    for class_id in classes_id:
        train_class_dir = os.path.join(TRAIN_BY_CLASS, class_id)
        validation_class_dir = os.path.join(VALIDATION_BY_CLASS, class_id)
        test_class_dir = os.path.join(TEST_BY_CLASS, class_id)

        if not os.path.exists(validation_class_dir):
            os.mkdir(validation_class_dir)  

        if not os.path.exists(test_class_dir):
            os.mkdir(test_class_dir)

        files = glob(os.path.join(train_class_dir, '*'))[:num_to_move]
        files_validation = files[:num_validation]
        files_test = files[num_validation:]

        for f in files_validation:
            shutil.move(f, validation_class_dir)

        for f in files_test:
            shutil.move(f, test_class_dir)


# show total classes & images for training & validation after seperation
def overview_dataset():
    train = glob(os.path.join(TRAIN_BY_CLASS, '*'))
    validation = glob(os.path.join(VALIDATION_BY_CLASS, '*'))
    test = glob(os.path.join(TEST_BY_CLASS, '*'))

    total_train = sum(len(os.listdir(d)) for d in train)
    total_validation = sum(len(os.listdir(d)) for d in validation)
    total_test = sum(len(os.listdir(d)) for d in validation)

    print('total train classes: ', len(train))
    print('total validation classes: ', len(validation))
    print('total test classes: ', len(test))
    print('total train images: ', total_train)
    print('total validation images: ', total_validation)
    print('total test images: ', total_test)


if __name__ == "__main__":
    # USE: python preprocess.py <min images> <num validation> <num test>
    min_img = int(sys.argv[1])
    num_validation = int(sys.argv[2])
    num_test = int(sys.argv[3])

    mkdir_by_class(remove_old=True)
    df_train = make_df_train()
    split_dataset(df_train, min_img, num_validation, num_test)
    overview_dataset()