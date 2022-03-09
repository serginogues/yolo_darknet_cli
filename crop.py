import argparse
import glob
from tqdm import tqdm
from PIL import Image


def crop(config):
    path = config.input_path
    label_id = config.id
    format = config.format
    output = config.output_path

    for image in tqdm(glob.glob(path + "*" + format), desc="Exporting crops"):
        # read image
        im = Image.open(image)
        im_w = im.width
        im_h = im.height

        # read label coordinates
        img_path = image.split(format)[0]
        im_name = img_path.split("\\")[-1]
        with open(img_path+'.txt') as f:
            lines = f.readlines()
            labels = []
            for line in lines:
                nums = line.split(" ")
                if int(nums[0]) == label_id:
                    x = float(nums[1])
                    y = float(nums[2])
                    w = float(nums[3])
                    h = float(nums[4])
                    left = int((x - (w/2))*im_w) - 50
                    upper = int((y - (h/2))*im_h) - 50
                    right = int((x + (w/2))*im_w) + 50
                    lower = int((y + (h/2))*im_h) + 50
                    labels.append((left, upper, right, lower))

        # YOLO labels: <object-class> <x> <y> <width> <height>
        # 4-tuple (left, upper),(right, lower)
        for c, box in enumerate(labels):
            im_crop = im.crop(box)
            im_crop.save(output + im_name + "_" + str(c) + ".jpeg", "JPEG")

    print("Cropped images exported at ", output)


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