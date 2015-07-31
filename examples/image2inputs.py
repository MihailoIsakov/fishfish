__author__ = 'zieghailo'

import os
import cv2
import numpy as np

def load_images():
    uri = '/home/zieghailo/development/projects/psiml/puffer/data/cutouts/'
    uri_no_puffs = 'no-puffs-resized/'
    uri_puffs = 'puffs-resized/'

    puff_files = os.listdir(uri+uri_puffs)
    no_puff_files = os.listdir(uri+uri_no_puffs)


    puff_images = []
    for file in puff_files:
        img = cv2.imread(uri + uri_puffs + file)
        img = np.swapaxes(img, 0, 2)
        img = img[np.newaxis, ...]
        puff_images.append(img)

    puff_images = np.concatenate(puff_images)

    no_puff_images = []
    for file in no_puff_files:
        img = cv2.imread(uri + uri_no_puffs + file)
        img = np.swapaxes(img, 0, 2)
        img = img[np.newaxis, ...]
        no_puff_images.append(img)

    no_puff_images = np.concatenate(no_puff_images)

    return (puff_images, no_puff_images)


def build_set(no_puff_images, puff_images):
    inputs = np.concatenate((no_puff_images, puff_images))
    outputs = np.concatenate((
        np.zeros(len(no_puff_images)),
        np.ones(len(puff_images))
    ))

    length = len(inputs)

    indices = np.array(range(length))
    np.random.shuffle(indices)

    shuffled_input = inputs[indices]
    shuffled_output = outputs[indices]

    train_inputs = shuffled_input[:length*3/5]
    validate_inputs = shuffled_input[length*3/5 : length*4/5]
    test_inputs = shuffled_input[length*4/5:]

    train_outputs = shuffled_output[:length*3/5]
    validate_outputs = shuffled_output[length*3/5 : length*4/5]
    test_outputs = shuffled_output[length*4/5 :]

    return train_inputs, train_outputs, validate_inputs, validate_outputs, test_inputs, test_outputs




