"""
Prints number of labels per class and number of images of a Dataset with .txt YOLO format
"""

import argparse
import glob
import re
import matplotlib.pyplot as plt


def count(config):
    path = config.input
    PLOT = config.plot
    num_clases = int(input("Write number of clases:"))
    count_instances = [0 for _ in range(num_clases)]
    count_images = [0 for _ in range(num_clases)]

    class_labels = []
    total_images = 0
    total_instances = 0
    for filename in glob.glob(path + '*.txt', recursive=True):
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
                    class_labels.append(re.sub(r'[^a-zA-Z]', '', line))
        sorted_unique_labels = list(set(labels_all))
        for l in sorted_unique_labels:
            count_images[l] += 1
    print()
    for idx, i in enumerate(range(num_clases)):
        print(str(idx) + ") " + class_labels[i])
        print(" - " + str(count_instances[i]) + " instances")
        print(" - " + str(count_images[i]) + " images")

    if PLOT:
        plt.subplot(121)
        plt.pie(count_instances, labels=class_labels, autopct='%1.1f%%', startangle=90)
        plt.title("Instances per class (" + str(total_instances) + ")")
        plt.axis('equal')
        plt.subplot(122)
        plt.pie(count_images, labels=class_labels, autopct='%1.1f%%', startangle=90)
        plt.title("Images per class (" + str(total_images) + ")")
        plt.axis('equal')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="C:/darknet-master/data/obj/",
                        help='path to folder with imgs and labels')
    parser.add_argument('--plot', type=bool, default="False")
    config = parser.parse_args()
    count(config)