from config import MODELS, os
from utils import get_cfg, get_weights


def auto_label_main(model_key):
    """
    Example:
    darknet.exe detector test data\obj.data
    cfg\custom-yolov4_496_train_all_classes.cfg
    backup\train\yolov4\train_all_classes\496_train_all_classes\custom-yolov4_496_train_all_classes_final.weights
    -thresh 0.25 -dont_show -save_labels < data/train.txt
    """
    command_base = "darknet.exe detector test data\obj.data "
    keyword = "train"
    if ('andenes' or 'train') in model_key.lower():
        keyword = "train"
    elif ('torno' or 'turnstile') in model_key.lower():
        keyword = "turnstiles"

    clear = lambda: os.system('cls')
    clear()
    print("AUTO LABELING: " + keyword.upper())

    # get cfg file path
    cfg_path, num_classes = get_cfg(keyword)

    # get backup path
    weights_path = get_weights(keyword)