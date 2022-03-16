"""
Darknet shell main configuration parameters
"""

import glob
import os
import shutil
from random import randint

OPTIONS_LIST = ["Auto-label images with trained model",
                "Test a trained model",
                "Train a new model",
                "Crop and export Yolo labels",
                "Count label instances and images in Dataset",
                "Image similarity in a Dataset",
                "Export images containing specific object"]

MODELS_LIST = ["Andenes", "Tornos", "OCR"]

AUTO_LABEL_CMD_INIT = "darknet.exe detector test data\obj.data "
AUTO_LABEL_CMD_END = " -thresh 0.25 -dont_show -save_labels < data/train.txt"

BASE_PATH = "C:\\darknet-master"
DATA_PATH = os.path.join(BASE_PATH, 'data')
DATASETS_PATH = os.path.join(DATA_PATH, 'datasets')
CFG_PATH = os.path.join(BASE_PATH, 'cfg')
CFG_CUSTOM_PATH = os.path.join(CFG_PATH, 'custom_models')
BACKUP_PATH = os.path.join(BASE_PATH, 'backup')

IMG_FORMAT_LIST = ['.jpg', '.jpeg', '.png']
