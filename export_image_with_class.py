import argparse
import glob
from tqdm import tqdm
from PIL import Image


def crop(config):
    path = config.input
    label_id = config.id
    format = config.format
    output = config.output

    for image in tqdm(glob.glob(path + "*" + format), desc="Finding images and exporting"):
        # read image
        im = Image.open(image)

        # read label coordinates
        img_path = image.split(format)[0]
        im_name = img_path.split("\\")[-1]
        contains = False
        with open(img_path+'.txt') as f:
            lines = f.readlines()
            for line in lines:
                nums = line.split(" ")
                if int(nums[0]) == label_id:
                    contains = True

        # save image if contains label
        if contains:
            im.save(output + im_name + ".jpeg", "JPEG")

    print("Images exported at ", output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="input/",
                        help='path to folder with imgs and labels')
    parser.add_argument('--output', type=str, default="output/",
                        help='path to folder with imgs and labels')
    parser.add_argument('--id', type=int, default=4,
                        help='Id of the class from which to extract the crops per image. Default 4 (license_plate)')
    parser.add_argument('--format', type=str, default=".jpg",
                        help='Image type: .png, .jpg')
    config = parser.parse_args()
    crop(config)