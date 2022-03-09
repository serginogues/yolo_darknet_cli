"""
Scan similar images in a Dataset which you might want to remove
"""
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
import glob
from joblib import Parallel, delayed
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp

output = "output/new_dataset/"
output_similar = "output/similar_image_pairs/"
os.makedirs(output, exist_ok=True)
os.makedirs(output_similar, exist_ok=True)  # succeeds even if directory exists.

input = "D:/ImotionAnalytics/Datasets/620_train_all_classes_darknet/train/"
type = ".jpg"


def main(config):
    input_path = config.input
    type = config.format
    NJOBS = config.jobs

    image_array = []
    count = 0
    names = []
    for filename in tqdm(glob.glob(input_path + '*' + type), desc="Comparing images"):

        # read image
        img0 = cv2.imread(filename)
        img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))

        # get name
        img_path = filename.split(type)[0]
        img_name = img_path.split("\\")[-1]
        names.append(img_name)

        # compare with already appended images
        # if similar to any do not append
        result = compare_Parallel(img, image_array, NJOBS)
        if not any(result):
            image_array.append(img)
            cv2.imwrite(output + img_name + type, img0)
        count += 1

    print("Original dataset size: " + str(count))
    print("New dataset size: " + str(len(image_array)))


def is_similar(a, b) -> bool:
    MSE = mse(a, b)
    RMSE = rmse(a, b)
    ERGAS = ergas(a, b)

    if MSE < 50 and RMSE < 20 and ERGAS < 5000:
        numpy_horizontal = np.hstack((a, b))
        cv2.imwrite(output_similar + str(np.random.randint(1000000)) + type, numpy_horizontal)
        return True
    else:
        return False


def compare_Parallel(img, image_array, NJOBS):
    return Parallel(n_jobs=NJOBS)(delayed(is_similar)(img, imm) for idx, imm in enumerate(image_array[-20:]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="input/",
                        help='path to folder with imgs and labels')
    parser.add_argument('--format', type=str, default=".jpg",
                        help='Image type: .png, .jpg')
    parser.add_argument('--jobs', type=int, default=12,
                        help='Processor Jobs')
    config = parser.parse_args()
    main(config)