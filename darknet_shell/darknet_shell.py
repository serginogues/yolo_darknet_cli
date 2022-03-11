from config import *
from utils import ask_user_option
from auto_label import auto_label_main


def main():
    print("Welcome to Darknet Shell :')")
    action_key = OPTIONS[ask_user_option(OPTIONS)]
    model_key = MODELS[ask_user_option(MODELS)]

    if 'auto' and 'label' in action_key.lower():
        auto_label_main(model_key)
    elif 'test' and 'video' in action_key.lower():
        # test video
        pass
    elif 'test' and 'image' in action_key.lower():
        # test image
        pass


if __name__ == '__main__':
    main()

