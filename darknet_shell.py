import glob
import os
import shutil
from random import randint

OPTIONS = ["Auto Labeling", "Test model on Video", "Test model on Image"]
MODELS = ["Andenes", "Tornos"]

AUTO_LABEL_CMD_INIT = "darknet.exe detector test data\obj.data "
AUTO_LABEL_CMD_END = " -thresh 0.25 -dont_show -save_labels < data/train.txt"


def auto_label_main(keyword):
    """
    Example:
    darknet.exe detector test data\obj.data
    cfg\custom-yolov4_496_train_all_classes.cfg
    backup\train\yolov4\train_all_classes\496_train_all_classes\custom-yolov4_496_train_all_classes_final.weights
    -thresh 0.25 -dont_show -save_labels < data/train.txt
    """

    print("############# AUTO-LABEL: " + keyword.upper() + " #############")

    # get cfg file path
    cfg_path, num_classes = get_cfg(keyword)

    # get backup path
    weights_path = get_weights(keyword)

    # dataset path
    dataset_path = get_dataset(False)

    # update train.txt and copy images to obj/
    list_images(dataset_path, copy_labels=False)

    # update obj.data
    update_obj_data(num_classes)

    # update obj.names
    print()
    print("You have " + str(num_classes)
          + " classes. Please update data/obj.names with correct labels (one label per line). Enter '0' once done.")
    ask_user_option(['Continue'])

    # execute command
    full_cmd = AUTO_LABEL_CMD_INIT + cfg_path + " " + weights_path + AUTO_LABEL_CMD_END

    clear = lambda: os.system('cls')
    clear()
    print("The following command will be executed. If it is correct enter '0' to begin auto-label:")
    print()
    print(full_cmd)
    ask_user_option(['Continue'])

    os.system(full_cmd)


def pick_model() -> str:
    model_key = MODELS[ask_user_option(MODELS)]
    keyword = "train"
    if ('andenes' or 'train') in model_key.lower():
        keyword = "train"
    elif ('torno' or 'turnstile') in model_key.lower():
        keyword = "turnstiles"

    return keyword


def main():
    print("Welcome to Darknet Shell :')")
    action_key = OPTIONS[ask_user_option(OPTIONS)]
    keyword = pick_model()
    clear = lambda: os.system('cls')
    clear()

    if 'auto' and 'label' in action_key.lower():
        auto_label_main(keyword)
    elif 'test' and 'video' in action_key.lower():
        pass
    elif 'test' and 'image' in action_key.lower():
        pass


def get_file_name_from_path(path):
    return path.split("\\")[-1]


def get_dataset(is_train_and_valid = True) -> str:
    DATASETS_PATH = "darknet_shell/data/datasets/"
    found_dataset = False
    dataset_path = ""
    while not found_dataset:
        dataset_path = choose_directory(DATASETS_PATH)
        has_train_and_valid = dataset_has_train_valid_subfolders(dataset_path)
        if has_train_and_valid != is_train_and_valid:
            print()
            print("valid/ and train/ not found (for TRAINING and VALIDATION) "
                  "OR valid/ and train/ found but not needed (AUTO-LABEL)")
        else:
            found_dataset = True

    print("############### dataset PATH = " + dataset_path)
    print()
    return dataset_path


def get_weights(model: str = "turnstiles"):
    path = "darknet_shell/backup/custom_models/"
    list_weights = []

    print("List of all available .weight filess:")
    print()
    for filename in glob.glob(path + '*.weights', recursive=True):
        count = 0
        if model in filename:
            cfg_name = get_file_name_from_path(filename)
            list_weights.append(filename)
            print("   " + str(count) + " - " + cfg_name)
            count += 1

    final_path = list_weights[ask_user_option(list_weights, print_options=False)]
    print("############### .weights PATH = " + final_path)
    print()
    return final_path


def get_cfg(model: str = "turnstiles"):
    path = "darknet_shell/cfg/custom_models/"
    num_classes = 0
    list_cfg = []

    print("List of all available .cfg files and number of classes:")
    print()
    for filename in glob.glob(path + '*.cfg', recursive=True):
        count = 0
        if model in filename:
            # read num classes
            with open(filename, 'r+') as out:
                lines = out.readlines()
                for line in lines:
                    if "classes" in line:
                        num_classes = int(line.split("=")[1])
                        break
            cfg_name = get_file_name_from_path(filename)
            list_cfg.append((filename, num_classes))
            print("   " + str(count) + " - " + cfg_name + " (" + str(num_classes) + " classes)")
            count += 1

    cfg_path, num_classes = list_cfg[ask_user_option([x for x,y in list_cfg], print_options=False)]
    print("############### .cfg PATH = " + cfg_path)
    print("############### Nº CLASSES = " + str(num_classes))
    print()

    return cfg_path, num_classes


def ask_user_option(params: list, print_options: bool = True) -> int:
    print()
    if print_options:
        for idx, opt in enumerate(params):
            print(str(idx) + " - " + opt)
    while True:
        try:
            action = int(input("Choose an option (write option number):"))
            if 0 <= action < len(params):
                print()
                return action
        except ValueError:
            continue


def ask_user_path() -> str:
    while True:
        try:
            action = input("Give path to dataset:")
            if type(action) is str:
                return action
        except ValueError:
            continue


def list_images(path_from: str, is_train: bool = True, copy_labels: bool = True):
    """
    Creates obj or valid folders, copies files from dataset and writes train.txt or valid.txt
    :param path_from: path to dataset where images are stored
    :param is_train: if True, creates obj/ and writes in train.txt, else valid/ and valid.txt
    """

    folder_to = "train" if is_train else "valid"
    path_to = 'data\\obj\\' if is_train else 'data\\valid\\'

    try:
        # create obj or valid
        if os.path.exists(path_to):
            shutil.rmtree(path_to)
        os.makedirs(path_to)
    except OSError:
        print("Creation of the directory %s failed" % path_to)
    else:
        print("############### %s created", path_to)
        print()

    # write in train.txt or valid.txt and copy images
    try:

        with open(os.path.join('darknet_shell/data', folder_to + '.txt'), 'w') as out:
            for img in [f for f in os.listdir(path_from + '\\') if (f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg'))]:
                out.write(path_to + img + '\n')  # write in txt
                shutil.copyfile(path_from + "\\" + img, path_to + img)  # copy images

                if copy_labels:
                    img_name = get_file_name_from_path(img)
                    shutil.copyfile(path_from + "\\" + img_name + ".txt", path_to + img_name + ".txt")  # copy txt labels
    except OSError:
        print("Error while copying images from (list_images()) %s", path_from)
    else:
        print("############### %s.txt written", folder_to)
        print("############### images added to %s.txt written", folder_to)
        print()


def choose_directory(path: str) -> str:
    """
    List available folders and return chosen by user
    """
    print()
    print("List of available folders:")
    list_subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    folder_path = list_subfolders[ask_user_option([get_file_name_from_path(x) for x in list_subfolders])]
    return folder_path


def dataset_has_train_valid_subfolders(path: str) -> bool:
    """
    :param path: path to dataset
    :return: True if repo contains 'train' and 'valid' folders
    """
    list_subfolders = [f.name for f in os.scandir(path) if f.is_dir()]
    has_train = False
    has_valid = False
    for path in list_subfolders:
        if 'train' in path:
            has_train = True
        elif 'valid' in path:
            has_valid = True

    return has_train and has_valid


def update_obj_data(num_classes: int, create_backup: bool = False):
    print()
    backup_path = "C:/darknet-master/backup/new_training_" + str(randint(0, 1000000))
    text = "classes = " + str(num_classes) + "\n" + "train = data/train.txt\n" \
           + "valid = data/valid.txt\n" + "names = data/obj.names\n" \
            + "backup = " + backup_path

    if create_backup:
        try:
            if os.path.exists(backup_path):
                shutil.rmtree(backup_path)
            os.makedirs(backup_path)
        except OSError:
            print("Creation of the directory %s failed" % backup_path)

    with open('darknet_shell/data/obj.data', 'r+') as myfile:
        data = myfile.read()
        myfile.seek(0)
        myfile.write(text)
        myfile.truncate()

    print("############### data/obj.data updated with following: \n")
    print(text)
    print()


if __name__ == '__main__':
    main()
