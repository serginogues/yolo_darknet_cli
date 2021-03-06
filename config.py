"""
Darknet shell main configuration parameters
"""

import glob
import os
import subprocess
import shutil
from random import randint
from tqdm import tqdm
import cv2
import numpy as np
from joblib import Parallel, delayed
from sewar.full_ref import mse, rmse, ergas
import multiprocessing
import matplotlib.pyplot as plt
import pandas as pd

OPTIONS_LIST = ["AUTO-LABEL (semi-supervised learning)",  # 0
                "TEST trained model on SINGLE image or video",  # 1
                "TRAIN a new model",  # 2
                "VALIDATE a trained model",  # 3
                "CROP labeled images",  # 4
                "COUNT label instances and images in Dataset",  # 5
                "Image SIMILARITY in a Dataset",  # 6
                "EXPORT images BY LABEL",  # 7
                "SAVE .weights every hour (while TRAIN)",  # 8
                "VALIDATE and COMPARE different models"]  # 9

MODELS_LIST = ["Andenes", "Tornos", "OCR"]

BASE_PATH = "C:\\darknet-master"

DATA_PATH = os.path.join(BASE_PATH, 'data')  # /data/
OBJ_DATA_FILE_PATH = os.path.join(DATA_PATH, 'obj.data')  # /data/obj.data
DATA_OBJ_PATH = os.path.join(DATA_PATH, 'obj')  # /data/obj/
DATA_VALID_PATH = os.path.join(DATA_PATH, 'valid')  # data/valid/

DATASETS_PATH = os.path.join(DATA_PATH, 'datasets')  # data/datasets/
DATASETS_CUSTOM_PATH = os.path.join(DATASETS_PATH, 'custom_models')  # data/datasets/custom_models/
DATASETS_OPEN_SOURCE_PATH = os.path.join(DATASETS_PATH, 'open_source_models')  # data/datasets/open_source_models/

CFG_PATH = os.path.join(BASE_PATH, 'cfg')  # /data/cfg/
BACKUP_PATH = os.path.join(BASE_PATH, 'backup')  # data/backup/

IMG_FORMAT_LIST = ['.jpg', '.jpeg', '.png']
