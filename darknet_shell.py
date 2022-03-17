"""
Darknet Shell MAINSCRIPT
"""

from config import *
from utils import *


def main():
    new_shell_section("Welcome to Darknet Shell :')")
    action_key = ask_user_option(OPTIONS_LIST)

    if action_key == OPTIONS_LIST[0]:
        # auto-label
        auto_label_main()
    elif action_key == OPTIONS_LIST[1]:
        # test
        pass
    elif action_key == OPTIONS_LIST[2]:
        # train
        pass
    elif action_key == OPTIONS_LIST[3]:
        # crop
        crop()
    elif action_key == OPTIONS_LIST[4]:
        # count labels
        count_labels()
    elif action_key == OPTIONS_LIST[5]:
        # img similarity
        pass
    elif action_key == OPTIONS_LIST[6]:
        # export by class
        export_image_with_given_label()


def get_model() -> str:
    new_shell_section("Provide MODEL")
    model_key = ask_user_option(MODELS_LIST)
    keyword = "train"
    if model_key == MODELS_LIST[0]:
        keyword = "train"
    elif model_key == MODELS_LIST[1]:
        keyword = "turnstiles"
    elif model_key == MODELS_LIST[2]:
        keyword = "ocr"
    return keyword


def auto_label_main():
    """
    Example:
    darknet.exe detector test data\obj.data
    cfg\custom-yolov4_496_train_all_classes.cfg
    backup\train\yolov4\train_all_classes\496_train_all_classes\custom-yolov4_496_train_all_classes_final.weights
    -thresh 0.25 -dont_show -save_labels < data/train.txt
    """

    dataset_path = get_dataset(False)
    classes, num_classes = get_classes(dataset_path)
    keyword = get_model()
    cfg_path = get_cfg(num_classes, keyword)
    weights_path = get_weights(keyword)
    copy_files_train_valid(dataset_path, OBJ=True, copy_labels=False)
    update_obj_data(classes, create_backup=False)

    # execute command
    full_cmd = DARKNET_EXE_PATH + "detector test " + OBJ_DATA_FILE_PATH + " " + cfg_path + " " + weights_path + AUTO_LABEL_CMD_END

    print("Execute the following command at " + BASE_PATH)
    print()
    print(full_cmd)
    os.chdir(BASE_PATH)
    ask_user_option(['Start'])
    os.system(full_cmd)


def crop():
    """
    parser.add_argument('--input', type=str, default="input/",
                        help='path to folder with imgs and labels')
    parser.add_argument('--output', type=str, default="output/",
                        help='path to folder with imgs and labels')
    parser.add_argument('--id', type=int, default=4,
                        help='Id of the class from which to extract the crops per image. Default 4 (license_plate)')
    parser.add_argument('--format', type=str, default=".jpg",
                        help='Image type: .png, .jpg')
    """
    from PIL import Image

    print("Provide path to folder with images and labels")
    path = ask_user_path()
    classes = get_classes(path)
    print("Enter desired class id to crop images:")
    label_id = ask_user_option(classes, return_idx=True)
    print("Provide output path:")
    output = ask_user_path()

    def crop_by_format(path: str, format: str):
        for image in tqdm(glob.glob(path + format), desc="Exporting crops for " + format):
            # read image
            im = Image.open(image)
            im_w = im.width
            im_h = im.height

            # read label coordinates
            img_path = image.split(format)[0]
            im_name = img_path.split("\\")[-1]
            with open(img_path + '.txt') as f:
                lines = f.readlines()
                labels = []
                for line in lines:
                    nums = line.split(" ")
                    if int(nums[0]) == label_id:
                        x = float(nums[1])
                        y = float(nums[2])
                        w = float(nums[3])
                        h = float(nums[4])
                        left = int((x - (w / 2)) * im_w) - 50
                        upper = int((y - (h / 2)) * im_h) - 50
                        right = int((x + (w / 2)) * im_w) + 50
                        lower = int((y + (h / 2)) * im_h) + 50
                        labels.append((left, upper, right, lower))

            # YOLO labels: <object-class> <x> <y> <width> <height>
            # 4-tuple (left, upper),(right, lower)
            for c, box in enumerate(labels):
                im_crop = im.crop(box)
                im_crop.save(output + im_name + "_" + str(c) + format, format.split(".")[1])

    for f in IMG_FORMAT_LIST:
        crop_by_format(path, f)
    print("Cropped images exported at ", output)


def count_labels():
    """
    Prints number of labels per class and number of images of a Dataset with .txt YOLO format \n
    parser.add_argument('--input', type=str, default="C:/darknet-master/data/obj/",
                        help='path to folder with imgs and labels')
    parser.add_argument('--plot', type=bool, default="False")
    """

    print("Would you like to plot results? (requires 'matplotlib' installed in current environment)")
    PLOT = True if ask_user_option(['Yes', 'No'], return_idx=True) == 0 else False

    print("Provide path to folder with images and labels")
    path = ask_user_path()

    if path[-1] != '\\':
        path = path + "\\"

    classes, num_classes = get_classes(path)

    count_instances = [0 for _ in range(num_classes)]
    count_images = [0 for _ in range(num_classes)]

    total_images = 0
    total_instances = 0
    for filename in tqdm(glob.glob(path + '*.txt', recursive=True), desc="Counting images"):
        labels_all = []
        total_images += 1
        with open(filename, 'r+') as out:
            lines = out.readlines()
            for line in lines:
                try:
                    clase = int(line.split(" ")[0])
                    labels_all.append(clase)
                    count_instances[clase] += 1
                    total_instances += 1
                except ValueError:
                    continue
        sorted_unique_labels = list(set(labels_all))
        for l in sorted_unique_labels:
            count_images[l] += 1
    print(" class id - # instances - # images")
    for i in range(num_classes):
        print(classes[i] + " - " + str(count_instances[i]) + " - " + str(count_images[i]))

    if PLOT:
        import matplotlib.pyplot as plt
        plt.subplot(121)
        plt.pie(count_instances, labels=classes, autopct='%1.1f%%', startangle=90)
        plt.title("Instances per class (" + str(total_instances) + ")")
        plt.axis('equal')
        plt.subplot(122)
        plt.pie(count_images, labels=classes, autopct='%1.1f%%', startangle=90)
        plt.title("Images per class (" + str(total_images) + ")")
        plt.axis('equal')
        plt.show()


def export_image_with_given_label():
    """
    parser.add_argument('--input', type=str, default="input/",
                        help='path to folder with imgs and labels')
    parser.add_argument('--output', type=str, default="output/",
                        help='path to folder with imgs and labels')
    parser.add_argument('--id', type=int, default=4,
                        help='Id of the class from which to extract the crops per image. Default 4 (license_plate)')
    parser.add_argument('--format', type=str, default=".jpg",
                        help='Image type: .png, .jpg')
    """
    from PIL import Image

    print("Provide path to folder with images and labels")
    path = ask_user_path()

    classes = get_classes(path)
    print("Enter desired class id to crop images")
    label_id = ask_user_option(classes, return_idx=True)
    print("Provide output path (it will be created if it does not exist)")
    print("remember to write path with '\' at the end")
    output = ask_user_path()
    output_contains = output + "contains_" + classes[label_id] + "\\"
    output_not_contains = output + "not_contains_" + classes[label_id] + "\\"

    try:
        # create obj or valid
        if os.path.exists(output_contains):
            shutil.rmtree(output_contains)
        else:
            os.makedirs(output_contains)
    except ValueError:
        pass

    try:
        # create obj or valid
        if os.path.exists(output_not_contains):
            shutil.rmtree(output_not_contains)
        else:
            os.makedirs(output_not_contains)
    except ValueError:
        pass

    def export_by_type(path: str, format: str):
        for image in tqdm(glob.glob(path + "*" + format), desc="Finding images and exporting for " + format):
            # read image
            im = Image.open(image)

            # read label coordinates
            img_path = image.split(format)[0]
            im_name = img_path.split("\\")[-1]
            contains = False
            with open(img_path + '.txt') as f:
                lines = f.readlines()
                for line in lines:
                    nums = line.split(" ")
                    if int(nums[0]) == label_id:
                        contains = True
                        break

            # save image if contains label
            if contains:
                im.save(output_contains + im_name + format, format.split(".")[-1])
            else:
                im.save(output_not_contains + im_name + format, format.split(".")[-1])

    for f in IMG_FORMAT_LIST:
        export_by_type(path, f)
    print("Images exported at ", output)


if __name__ == '__main__':
    main()
