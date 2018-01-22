from os import listdir
from os.path import isfile, join
from random import randint
import shutil

def move_image(file_name, split, category):
    shutil.move(category + '/' + file_name, split + '/' + category + '/' + file_name)

def split_dataset_for_category(category):
    images = [f for f in listdir(category) if isfile(join(category, f)) and f.endswith('.png')]

    for image in images:
        r = randint(0, 9)

        if r < 7:
            move_image(image, 'train', category)
        elif r == 9:
            move_image(image, 'valid', category)
        else:
            move_image(image, 'test', category)

split_dataset_for_category('cardiomegaly')
split_dataset_for_category('no finding')

