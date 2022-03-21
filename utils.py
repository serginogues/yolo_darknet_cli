"""
Generic small blocks that can be re-used
"""

from config import *


def get_file_name_from_path(path, element: int = 1):
    """
    Given a full path to a file, returns the name of that file including format (i.e. 'name_of_file.txt')
    """
    return path.split("\\")[-element]


def ask_user_option(params: list, print_options: bool = True, return_idx: bool = False):
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
    folder_path = ask_user_option(list_subfolders)
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
    new_shell_section("Provide DATASET")
    found_dataset = False
    dataset_path = ""
    while not found_dataset:
        print("Where is your dataset?")
        pathh = ask_user_option([DATASETS_CUSTOM_PATH, DATASETS_OPEN_SOURCE_PATH])
        dataset_path = choose_directory(pathh)
        has_train_and_valid = dataset_has_train_valid_subfolders(dataset_path)
        if has_train_and_valid and not requires_train_and_valid:
            print("Wrong dataset :(")
            print(dataset_path + " contains train/ and valid/ sub-folders.")
            print("Please choose another dataset with all images and no sub-folders.")
        elif not has_train_and_valid and requires_train_and_valid:
            print(dataset_path + " does not have train/ and valid/ sub-folders.")
        else:
            found_dataset = True
    return dataset_path


def get_weights(model: str = "turnstiles"):
    """
    :param model: YOLO model
    :return: path to .weights
    """
    new_shell_section("Provide WEIGHTS file")
    list_weights = []
    for root, dirs, files in os.walk(BACKUP_PATH):
        for file in files:
            if file.endswith(".weights") and model in file:
                filename = os.path.join(root, file)
                list_weights.append(filename)

    final_path = ask_user_option(list_weights)
    return final_path


def get_cfg(num_classes: int, model: str = "turnstiles"):
    """
    :param num_classes: number of classes calculated when reading classes.txt or darknet.labels.
    cfg files with different number of classes won't be suggested.
    :param model: YOLO model
    :return: path to .cfg
    """
    new_shell_section("Provide CFG file")
    print('Please note that .cfg files with different number of classes than ' + str(num_classes) + ' are not shown')
    list_cfg = []
    for root, dirs, files in os.walk(CFG_PATH):
        for file in files:
            if file.endswith(".cfg") and model in file:
                filename = os.path.join(root, file)
                if get_num_classes_from_cfg(filename) == num_classes:
                    list_cfg.append(filename)

    cfg_path = ask_user_option(list_cfg)
    return cfg_path


def get_num_classes_from_cfg(cfg_path: str) -> int:
    with open(cfg_path, 'r+') as out:
        lines = out.readlines()
        for line in lines:
            if "classes" in line:
                num_classes = int(line.split("=")[1])
                return num_classes


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


def get_classes(path: str) -> (list, int):
    """
    :param path: path to dataset
    :return: list of labels
    """
    new_shell_section("Provide LABELS")

    # look for existing classes.txt or labels.darknet at given path
    files = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in [f for f in filenames if (f == "classes.txt" or f == "_darknet.labels")]:
            files.append(os.path.join(dirpath, filename))

    if len(files) > 0:
        classes = read_classes(files[0])
        print(classes)
        return classes, len(classes)
    else:
        # propose existing files in data/datasets/
        print("classes.txt and _darknet.labels not found at", path)
        files = []
        for dirpath, dirnames, filenames in os.walk(DATASETS_PATH):
            for filename in [f for f in filenames if (f == "classes.txt" or f == "_darknet.labels")]:
                files.append(os.path.join(dirpath, filename))
        classes = read_classes(files[ask_user_option(
            ["../" + get_file_name_from_path(x, element=3) + "/: " + ", ".join(read_classes(x)) for x in files],
            return_idx=True)])
        print(classes)
        return classes, len(classes)


def read_classes(file: str) -> list:
    classes = []
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            new = line.replace('\n', '')
            new = new.replace('\t', '')
            classes.append(new)
    return classes


def copy_files_and_write_path(path_dataset: str, path_to: str,  path_write: str, copy_labels: bool):
    """
    Do not add \\ at the end of any path
    :param path_dataset: dataset where images are stored
    :param path_to: path to folder where images are copied to
    :param path_write: path to txt file where image paths are written
    :param copy_labels: if True, path_dataset/ contains labels
    """
    new_shell_section("Copying images from: " + path_dataset)
    print("To: " + path_to)
    print("Copying label files (.txt) ? " + "Yes" if copy_labels else "No")
    print("Writting paths at: " + path_write)

    try:
        if os.path.exists(path_to):
            shutil.rmtree(path_to)
        os.makedirs(path_to)
    except OSError:
        print("Creation of the directory %s failed" % path_to)

    with open(path_write, 'w') as out:
        for img in tqdm([f for f in os.listdir(path_dataset + '\\') if
                         (f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg'))],
                        desc="Copying files"):

            # img = name.png
            image_path = os.path.join(path_to, img)
            out.write(image_path + '\n')  # write in txt
            shutil.copyfile(os.path.join(path_dataset, img), image_path)  # copy images

            if copy_labels:
                image_labels = img.split('.')[0] + ".txt"
                shutil.copyfile(os.path.join(path_dataset, image_labels), os.path.join(path_to, image_labels))


def update_obj_data(classes: list, create_backup: bool = False):
    """
    - Update obj.data \n
    - Update obj.names and coco.names \n
    - Create backup (if 'create_backup' = True)
    """
    new_shell_section("Writing ../data/obj.data, ../data/obj.names, and ../data/coco.names")
    backup_path = "C:/darknet-master/backup/new_training_" + str(randint(0, 1000000))

    update_file(OBJ_DATA_FILE_PATH, ["classes = " + str(len(classes)),
                                                      "train = data/train.txt",
                                                      "valid = data/valid.txt",
                                                      "names = data/obj.names",
                                                      "backup = " + backup_path])
    update_file(os.path.join(DATA_PATH, 'obj.names'), classes)
    update_file(os.path.join(DATA_PATH, 'coco.names'), classes)

    if create_backup:
        try:
            if os.path.exists(backup_path):
                shutil.rmtree(backup_path)
            os.makedirs(backup_path)
        except OSError:
            print("Creation of the directory %s failed" % backup_path)


def update_file(path: str, line_list: list):
    """
    :param path:
    :param line_list: each line to be written to file
    :return:
    """
    open(path, 'w').close()
    with open(path, 'w') as file:
        for listitem in line_list:
            file.write('%s\n' % listitem)


def new_shell_section(TITLE: str):
    os.system('cls' if os.name == 'nt' else 'clear')
    print(TITLE)
