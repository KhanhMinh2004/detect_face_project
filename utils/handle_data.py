import os
import numpy as np

from PIL import Image


dict = {
    'train_Minh': [ 1, 0, 0],
    'train_NChau': [0, 1, 0],
    'train_TVy': [ 0, 0, 1],
    'test_Minh': [ 1, 0, 0],
    'test_NChau': [ 0, 1, 0],
    'test_TVy': [ 0, 0, 1]
}

def get_data(dir_data, lst_data):

    for whatelse in os.listdir(dir_data):
        whatelse_path = os.path.join(dir_data, whatelse)
        lst_img_path = []

        for sub_whatelse in os.listdir(whatelse_path):
            img_path = os.path.join(whatelse_path, sub_whatelse)
            label = img_path.split('\\')[1]
            img = np.array(Image.open(img_path))
            lst_img_path.append((img, dict[label]))
        lst_data.extend(lst_img_path)

    return lst_data
