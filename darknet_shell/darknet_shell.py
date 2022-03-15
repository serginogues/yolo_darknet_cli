from utils import *

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


if __name__ == '__main__':
    main()

