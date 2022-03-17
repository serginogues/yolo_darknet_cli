"""
Darknet shell main configuration parameters
"""

import glob
import os
import shutil
from random import randint
from tqdm import tqdm

OPTIONS_LIST = ["Auto-label images with trained model",
                "Test a trained model",
                "Train a new model",
                "Crop and export Yolo labels",
                "Count label instances and images in Dataset",
                "Image similarity in a Dataset",
                "Export images containing specific object"]

MODELS_LIST = ["Andenes", "Tornos", "OCR"]

DARKNET_EXE_PATH = "C:\darknet-master\darknet.exe "
AUTO_LABEL_CMD_END = " -thresh 0.25 -dont_show -save_labels < data/train.txt"

BASE_PATH = "C:\\darknet-master"
DATA_PATH = os.path.join(BASE_PATH, 'data')
DATA_OBJ_PATH = os.path.join(DATA_PATH, 'obj')
OBJ_DATA_FILE_PATH = os.path.join(DATA_PATH, 'obj.data')
DATA_VALID_PATH = os.path.join(DATA_PATH, 'valid')
DATASETS_PATH = os.path.join(DATA_PATH, 'datasets')
DATASETS_CUSTOM_PATH = os.path.join(DATASETS_PATH, 'custom_models')
DATASETS_OPEN_SOURCE_PATH = os.path.join(DATASETS_PATH, 'open_source_models')
CFG_PATH = os.path.join(BASE_PATH, 'cfg')
BACKUP_PATH = os.path.join(BASE_PATH, 'backup')

IMG_FORMAT_LIST = ['.jpg', '.jpeg', '.png']
