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
        test_main()
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
        img_similarity()
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
    cfg\..\name.cfg
    backup\..\name.weights
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
    full_cmd = "darknet.exe detector test " + OBJ_DATA_FILE_PATH \
               + " " + cfg_path + " " + weights_path \
               + " -thresh 0.25 -dont_show -save_labels < data/train.txt"

    print()
    print("Labels will be created at ../data/obj/")
    print("Command: ")
    print(full_cmd)
    print()
    print("Enter '0' to start")
    os.chdir(BASE_PATH)
    ask_user_option(['Start'])
    os.system(full_cmd)


def test_main():
    new_shell_section("Provide path to image or video including file extension .png, .jpg, .jpeg, .mp4, etc")
    path = ask_user_path()
    classes, num_classes = get_classes(path)
    keyword = get_model()
    cfg_path = get_cfg(num_classes, keyword)
    weights_path = get_weights(keyword)
    update_file(os.path.join(DATA_PATH, 'obj.names'), classes)
    update_file(os.path.join(DATA_PATH, 'coco.names'), classes)

    extension = "." + path.split(".")[-1]
    isIMAGE = True if extension in IMG_FORMAT_LIST else False
    if isIMAGE:
        full_cmd = "darknet.exe detector test " + OBJ_DATA_FILE_PATH \
                   + " " + cfg_path + " " + weights_path \
                   + " " + path

    else:
        full_cmd = "darknet.exe detector demo " + OBJ_DATA_FILE_PATH \
                   + " " + cfg_path + " " + weights_path \
                   + " " + path + " -out_filename " \
                   + os.path.join("\\".join(path.split("\\")[0:-1]),
                     ".".join(get_file_name_from_path(path).split(".")[0:-1]) + "_output" + extension)

    print(full_cmd)
    os.chdir(BASE_PATH)
    os.system(full_cmd)


def crop():
    """
    parser.add_argument('--input', type=str, default="input/",
                        help='path to folder with imgs and labels')
    parser.add_argument('--output', type=str, default="output/",
                        help='path to folder with imgs and labels')
    parser.add_argument('--id', type=int, default=4,
                        help='Id of the class from which to extract the crops per image.')
    parser.add_argument('--format', type=str, default=".jpg",
                        help='Image type: .png, .jpg')
    """
    from PIL import Image

    print("Provide path to folder with images and labels")
    path = ask_user_path()
    if path[-1] != '\\':
        path = path + "\\"

    classes, num_classes = get_classes(path)
    print("Enter desired class id to crop images:")
    label_id = ask_user_option(classes, return_idx=True)
    print("Provide output path:")
    output = ask_user_path()
    if output[-1] != '\\':
        output = output + "\\"

    def crop_by_format(path: str, format: str):
        for image in tqdm(glob.glob(path + "*" + format), desc="Exporting crops for " + format):
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
                im_crop.save(output + im_name + "_" + str(c) + format)

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

    print("Would you like to plot results?")
    PLOT = True if ask_user_option(['Yes', 'No'], return_idx=True) == 0 else False

    print("Provide path to folder with images and labels (no need to add \ at the end)")
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
                        help='Id of the class from which to extract the crops per image.')
    parser.add_argument('--format', type=str, default=".jpg",
                        help='Image type: .png, .jpg')
    """
    from PIL import Image

    print("Provide path to folder with images and labels")
    path = ask_user_path()

    if path[-1] != '\\':
        path = path + "\\"

    classes, num_classes = get_classes(path)
    print("Enter desired class id to crop images")
    label_id = ask_user_option(classes, return_idx=True)
    print("Provide output path (it will be created if it does not exist)")
    output = ask_user_path()
    if output[-1] != '\\':
        output = output + "\\"
    output_contains = output + "contains_" + str(label_id) + "\\"
    output_not_contains = output + "not_contains_" + str(label_id) + "\\"

    try:
        # create obj or valid
        if os.path.exists(output_contains):
            shutil.rmtree(output_contains)
        else:
            os.makedirs(output_contains)
    except ValueError:
        print("Could not create directory")

    try:
        # create obj or valid
        if os.path.exists(output_not_contains):
            shutil.rmtree(output_not_contains)
        else:
            os.makedirs(output_not_contains)
    except ValueError:
        print("Could not create directory")

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
                im.save(output_contains + im_name + format)
                shutil.copyfile(img_path + ".txt", output_contains + im_name + ".txt")
            else:
                im.save(output_not_contains + im_name + format)
                shutil.copyfile(img_path + ".txt", output_not_contains + im_name + ".txt")

    for f in IMG_FORMAT_LIST:
        export_by_type(path, f)
    print("Images exported at ", output)


def img_similarity():
    """
    parser.add_argument('--input', type=str, default="input/",
                        help='path to folder with imgs and labels')
    parser.add_argument('--format', type=str, default=".jpg",
                        help='Image type: .png, .jpg')
    parser.add_argument('--jobs', type=int, default=12,
                        help='Processor Jobs')
    Scan similar images in a Dataset which you might want to remove
    """
    SIM_THRESHOLD_TRAIN = [50, 20, 5000]
    SIM_THRESHOLD_OCR = [1000, 100, 40000]  # 50, 20, 5000
    SIM_THRESHOLD_WHEELCHAIR = [8000, 800, 220000]
    THRESHOLD = [SIM_THRESHOLD_TRAIN, SIM_THRESHOLD_OCR, SIM_THRESHOLD_WHEELCHAIR]

    print("Provide path to folder with images and labels")
    input_path = ask_user_path()
    if input_path[-1] != '\\':
        input_path = input_path + "\\"

    new_shell_section(" ")

    output = input_path + "new_dataset/"
    output_similar = input_path + "similar_image_pairs/"
    os.makedirs(output, exist_ok=True)
    os.makedirs(output_similar, exist_ok=True)  # succeeds even if directory exists.
    print("Created ", output)
    print("Created ", output_similar)

    NJOBS = multiprocessing.Pool()._processes
    print("Found " + str(NJOBS) + " workers available in current pool.")

    print("Does your dataset contain labels (.txt files for each image)?")
    COPY_TXT = True if ask_user_option(['Yes', 'No'], return_idx=True) == 0 else False

    print("What similarity threshold would you like to use?")
    idx = ask_user_option(['ANDENES', 'OCR', 'WHEELCHAIR', 'Provide my own values'], return_idx=True)
    if idx < len(THRESHOLD):
        thr = THRESHOLD[idx]
    else:
        aa = int(input("MSE Threshold (write integer): "))
        bb = int(input("RMSE Threshold (write integer): "))
        cc = int(input("ERGAS Threshold (write integer): "))
        thr = [aa, bb, cc]

    def is_similar(a, b) -> bool:
        MSE = mse(a, b)
        RMSE = rmse(a, b)
        ERGAS = ergas(a, b)

        if MSE < thr[0] and RMSE < thr[1] and ERGAS < thr[2]:
            numpy_horizontal = np.hstack((a, b))
            cv2.imwrite(output_similar + str(np.random.randint(1000000)) + ".jpg", numpy_horizontal)
            return True
        else:
            return False

    def compare_Parallel(img, image_array, NJOBS):
        return Parallel(n_jobs=NJOBS)(delayed(is_similar)(img, imm) for idx, imm in enumerate(image_array[-20:]))

    image_array = []

    def run_by_type(type: str):
        for filename in tqdm(glob.glob(input_path + '*' + type), desc="Comparing images"):
            # read image
            img0 = cv2.imread(filename)
            img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))

            # get name
            img_path = filename.split(type)[0]
            img_name = img_path.split("\\")[-1]

            # compare with already appended images
            # if similar to any do not append
            result = compare_Parallel(img, image_array, NJOBS)
            if not any(result):
                image_array.append(img)
                cv2.imwrite(output + img_name + type, img0)
                if COPY_TXT:
                    shutil.copyfile(img_path + ".txt", output + img_name + ".txt")

    for f in IMG_FORMAT_LIST:
        run_by_type(f)
    print("New dataset size: " + str(len(image_array)))


if __name__ == '__main__':
    main()
