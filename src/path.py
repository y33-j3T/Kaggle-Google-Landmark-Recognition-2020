import os

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

# logs
CALLBACK_DIR = "logs"
MODELS_DIR = "models"