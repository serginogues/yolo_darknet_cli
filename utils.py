"""
Generic small blocks that can be re-used
"""

from config import *


def get_file_name_from_path(path):
    """
    Given a full path to a file, returns the name of that file including format (i.e. 'name_of_file.txt')
    """
    return path.split("\\")[-1]


def ask_user_option(params: list, print_options: bool = True, return_idx: bool = False) -> int:
    """
    :param return_idx: if True, returns index of chosen element in params, else returns element
    :param params: list of options from which the user has to choose
    :param print_options: if True, prints params
    :return: chosen element (or index if return_idx = True)
    """
    print()
    if print_options:
        for idx, opt in enumerate(params):
            print(str(idx) + " - " + opt)
    while True:
        try:
            action = int(input("Choose an option (write option number):"))
            if 0 <= action < len(params):
                print()

                if return_idx:
                    return action
                else:
                    return params[action]
        except ValueError:
            continue


def choose_directory(path: str) -> str:
    """
    Given a path, user can choose from the available sub-folders
    :param path: path to directory with sub-folders
    :return: path to chosen sub-folder
    """
    list_subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    folder_path = ask_user_option([get_file_name_from_path(x) for x in list_subfolders])
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


def get_dataset(requires_train_and_valid=True) -> str:
    """
    User picks from available datasets.
    :param requires_train_and_valid: True means that the dataset repo must
                                    have train/ and valid/ sub-folders (i.e. for training)
    :return: path to dataset
    """
    found_dataset = False
    dataset_path = ""
    while not found_dataset:
        dataset_path = choose_directory(DATASETS_PATH)
        has_train_and_valid = dataset_has_train_valid_subfolders(dataset_path)
        if has_train_and_valid != requires_train_and_valid:
            print("valid/ and train/ not found (for TRAINING and VALIDATION) "
                  "OR valid/ and train/ found but not needed (AUTO-LABEL)")
        else:
            found_dataset = True
    return dataset_path


def get_weights(model: str = "turnstiles"):
    """
    :param model: YOLO model
    :return: path to .weights
    """
    list_weights = []
    for filename in glob.glob(BACKUP_PATH + "\\" + '*.weights', recursive=True):
        count = 0
        if model in filename:
            cfg_name = get_file_name_from_path(filename)
            list_weights.append(filename)
            print("   " + str(count) + " - " + cfg_name)
            count += 1

    final_path = ask_user_option(list_weights, print_options=False)
    return final_path


def get_cfg(model: str = "turnstiles"):
    """
    :param model: YOLO model
    :return: path to .cfg
    """

    num_classes = 0
    list_cfg = []

    for filename in glob.glob(CFG_CUSTOM_PATH + "\\" + '*.cfg', recursive=True):
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

    cfg_path, num_classes = ask_user_option([x for x, y in list_cfg], print_options=False)
    return cfg_path, num_classes


def ask_user_path() -> str:
    """
    Asks user for path and checks that it exists
    :return: path provided by user
    """
    while True:
        try:
            path = input("Provide path:")
            if os.path.exists(path):
                return path
        except Exception:
            continue


def get_classes(path: str) -> list:
    """
    :param path: path to dataset
    :return: list of labels
    """
    files = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in [f for f in filenames if (f == "classes.txt" or f == "_darknet.labels")]:
            files.append(os.path.join(dirpath, filename))

    if len(files) > 0:
        classes = read_classes(files[0])
        return classes
    else:
        print("classes.txt and _darknet.labels not found.")
        print("Provide path to 'classes.txt' or '_darknet.labels':")
        while True:
            path2 = ask_user_path()
            if "classes.txt" in path2 or "_darknet.labels" in path2:
                classes = read_classes(path2)
                return classes
            else:
                print("Wrong file name provided.")
                continue


def read_classes(file: str) -> list:
    classes = []
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            new = line.replace('\n', '')
            new = new.replace('\t', '')
            classes.append(new)
    return classes


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
            for img in [f for f in os.listdir(path_from + '\\') if
                        (f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg'))]:
                out.write(path_to + img + '\n')  # write in txt
                shutil.copyfile(path_from + "\\" + img, path_to + img)  # copy images

                if copy_labels:
                    img_name = get_file_name_from_path(img)
                    shutil.copyfile(path_from + "\\" + img_name + ".txt",
                                    path_to + img_name + ".txt")  # copy txt labels
    except OSError:
        print("Error while copying images from (list_images()) %s", path_from)
    else:
        print("############### %s.txt written", folder_to)
        print("############### images added to %s.txt written", folder_to)
        print()


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