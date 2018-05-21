import sys

RUN_ID = sys.argv[1]
MODEL_DIR = '../Models/'

# train, validation and test dirs
TRAIN_DIR = '../Data/Images/Damage_Classification/training'
VAL_DIR = '../Data/Images/Damage_Classification/validation'
TEST_DIR = '../Data/Images/Damage_Classification/test'

TRAIN_STANFORD_DIR = '../Data/Images/Stanford_Dataset/training'
VAL_STANFORD_DIR = '../Data/Images/Stanford_Dataset/validation'
TEST_STANFORD_DIR = '../Data/Images/Stanford_Dataset/test'

# model hyper_parameters
# NUM_EPOCHS = int(sys.argv[2])
# LR_RATE = float(sys.argv[3])
# FINE_TUNE_LAYER = int(sys.argv[4])
# BATCH_SIZE = int(sys.argv[5])
# WEIGHT_DECAY = int(sys.argv[6])

NUM_EPOCHS = 10
LR_RATE = 1e-4
FINE_TUNE_LAYER = 13
BATCH_SIZE = 32
WEIGHT_DECAY = 0.0
