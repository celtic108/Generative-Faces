from PIL import Image
from os import walk, path
from tqdm import tqdm
import numpy as np


def load_image(filename):
    img = Image.open(filename).convert('RGB')
    img = np.array(img).astype('float32')
    img = (img / 127.5) - 1.0
    return img

def unpreprocess_image(image):
    img = (img + 1.0) * 255.
    img = img.astype('uint8')
    return img

def display_image(image):
    if image.shape[0] == 1:
        image = image[0]
    image = Image.fromarray(np.clip(image, 0, 255))
    image.show()

def get_image_file_list(folder_path):
    file_list = []
    for root, dirs, files in walk(folder_path):
        for file in files:
            if file.split('.')[-1] in ['jpeg', 'png', 'jpg', 'bmp', 'tif', 'tiff']:
                file_list.append(path.join(root, file))
    return file_list

def get_image_dataset(file_list):
    img_array = []
    for file in tqdm(file_list):
        img_array.append(load_image(file))
    img_array = np.array(img_array)
    return img_array
