"""
Darknet shell main configuration parameters
"""

import glob
import os
import shutil
from random import randint
from tqdm import tqdm
import cv2
import numpy as np
from joblib import Parallel, delayed
from sewar.full_ref import mse, rmse, ergas
import multiprocessing

OPTIONS_LIST = ["AUTO-LABEL (semi-supervised learning)",  # 0
                "TEST trained model on SINGLE image or video", # 1
                "TRAIN a new model", # 2
                "VALIDATE a trained model" # 3
                "CROP labeled images", # 4
                "COUNT label instances and images in Dataset", # 5
                "Image SIMILARITY in a Dataset", # 6
                "EXPORT images BY LABEL"] # 7

MODELS_LIST = ["Andenes", "Tornos", "OCR"]

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
