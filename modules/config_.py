# global params
COMMS_ROUND = 1000  # 800
# GLOBAL_LR = 0.01  # 0.02/60% (iid), 0.01/68% (iid), 0.01/34% (niid-1), 0.01/48% (niid-1 with ext)
PERC_CLIENT_SAMPLES = 1  # fraction of client to pick per round
NIID_NESS = 1  # Todo: tune
NUM_TOTAL_CLIENTS = 10  # Todo: tune
NUM_CLASSES = 10
MODEL_TYPE = 'lenet'
# GLR_DECAY_RATE = 0.95
# GLR_DECAY = False
# GLR_DECAY_STEPS = 100

# local params
LOCAL_LR = 0.07
LOCAL_MOMENTUM = 0.6
# LR_DECAY = 1
EXT_DECAY_RATE = 0.005  # 0.1/(900/50) = 0.005
START_EXT_DECAY_AFTER = 100
LOCAL_STEPS = 100
LOCAL_BS = 64  # 32
INIT_WEIGHT_PATH = None
PERC_DATA_EXTENSION = 0.1
UNBIASED_STEP_BS = 100
NUM_UNBIASED_STEPS = 1  # 10
FRAC_OF_UNBIASED_GRAD = 0.5  # 0.5
LR_DECAY_STEPS = 50
LR_DECAY_RATE = 0.9

# local params bools
DATA_EXTENSION = True
USE_UNBIASED_GRAD = False
EXT_DECAY = False

# type of extension
USE_PSEUDO_PATTERNS = False
USE_DATA_SHARING = True

RANDOM_EXT_LABEL = False
USE_DATA_AUGMENTATION = False
USE_LR_DECAY = False

# other params
PRIMARY_TRAIN_FEATURE_PATH = '/home/stijani/projects/dataset/cifar10/cifar10_32x32_ch_last_numpy/train_features.npy'
PRIMARY_TRAIN_LABEL_PATH = '/home/stijani/projects/dataset/cifar10/cifar10_32x32_ch_last_numpy/train_labels.npy'
PRIMARY_TEST_FEATURE_PATH = '/home/stijani/projects/dataset/cifar10/cifar10_32x32_ch_last_numpy/test_features.npy'
PRIMARY_TEST_LABEL_PATH = '/home/stijani/projects/dataset/cifar10/cifar10_32x32_ch_last_numpy/test_labels.npy'
METRIC_FILE_ACC = '/home/stijani/projects/phd/fed-learn/results/cifar10/tf/niid1/acc/with_data_sharing_data.csv'
METRIC_FILE_LOSS = '/home/stijani/projects/phd/fed-learn/results/cifar10/tf/niid1/loss/with_data_sharing_data.csv'
METRIC_DESC = 'FedAvg+10%DataSharing'
DATASET_NAME = 'CIFAR10'
DATA_PTH = '/home/stijani/projects/dataset/cifar10/'
EXT_FEATURE_PTH = '/home/stijani/projects/dataset/tiny-imagenet/tinyimagenet_32x32_ch_last_np/train_features.npy'
EXT_LABEL_PATH = '/home/stijani/projects/dataset/tiny-imagenet/tinyimagenet_32x32_ch_last_np/train_labels.npy'
