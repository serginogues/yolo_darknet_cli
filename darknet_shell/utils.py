from config import *


def get_weights(model: str = "turnstiles"):
    print()
    path = "backup/custom_models/"
    list_weights = []

    print("List of all available .weight filess:")
    print()
    for filename in glob.glob(path + '*.weights', recursive=True):
        count = 0
        if model in filename:
            cfg_name = filename.split("\\")[-1]
            list_weights.append(filename)
            print("   " + str(count) + " - " + cfg_name)
            count += 1

    return list_weights[ask_user_option(list_weights, print_options=False)]


def get_cfg(model: str = "turnstiles"):
    print()
    path = "cfg/custom_models/"
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
            cfg_name = filename.split("\\")[-1]
            list_cfg.append((filename, num_classes))
            print("   " + str(count) + " - " + cfg_name + " (" + str(num_classes) + " classes)")
            count += 1

    return list_cfg[ask_user_option([x for x,y in list_cfg], print_options=False)]


def ask_user_option(params: list, print_options: bool = True) -> int:
    print()
    if print_options:
        for idx, opt in enumerate(params):
            print(str(idx) + " - " + opt)
    while True:
        try:
            action = int(input("Choose an option (write option number):"))
            if 0 <= action < len(params):
                print("Chosen option: " + params[action])
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


def list_images(is_train: bool = True):
    file = "train" if is_train else "valid"
    with open('data/' + file + '.txt', 'w') as out:
        for img in [f for f in os.listdir('C:/darknet-master/data/' + file) if
                    not (f.endswith('.txt') or f.endswith('.labels'))]:
            out.write('C:/darknet-master/data/valid/' + img + '\n')


def choose_directory():
    """
    List available folders and return chosen by user
    """
    print("List of available folders:")
    pass