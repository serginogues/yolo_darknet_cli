"""
Prints number of labels per class and number of images of a Dataset with .txt YOLO format
"""

import argparse
import glob


def count(config):
    path = config.input
    num_clases = int(input("Write number of clases:"))
    count_clases = [0 for _ in range(num_clases)]
    count_images = [0 for _ in range(num_clases)]

    class_labels = []
    for filename in glob.glob(path + '*.txt', recursive=True):
        labels_all = []
        with open(filename, 'r+') as out:
            lines = out.readlines()
            for line in lines:
                try:
                    clase = int(line.split(" ")[0])
                    labels_all.append(clase)
                    count_clases[clase] += 1
                except ValueError:
                    class_labels.append(line)
        sorted_unique_labels = list(set(labels_all))
        for l in sorted_unique_labels:
            count_images[l] += 1
    print()
    for i in range(num_clases):
        print(class_labels[i].split("\\")[0])
        print(" - " + str(count_clases[i]) + " instances")
        print(" - " + str(count_images[i]) + " images")
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="C:/darknet-master/data/obj/",
                        help='path to folder with imgs and labels')
    config = parser.parse_args()
    count(config)