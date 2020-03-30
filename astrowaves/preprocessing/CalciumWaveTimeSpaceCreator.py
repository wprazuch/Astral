import os
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import re

logging.basicConfig(level=logging.INFO)


class CalciumWaveTimeSpaceCreator():

    def __init__(self, path_to_image_sequence=None):
        self.path_to_image_sequence = path_to_image_sequence

    def run(self, path_to_image_sequence=None):

        if path_to_image_sequence is not None:
            self.path_to_image_sequence = path_to_image_sequence

        slices = [slic for slic in os.listdir(path_to_image_sequence) if slic.endswith('.tif') and not slic.startswith('.')]
        slices.sort(key=lambda f: int(re.sub('\D', '', f)))
        no_slices = len(slices)
        logging.info("Detected {} image slices".format(no_slices))
        I = plt.imread(os.path.join(path_to_image_sequence, slices[no_slices-1]))
        img_shape = I.shape
        logging.info("Detected image shape of {} pixels".format(img_shape))
        image_matrix = np.ndarray(shape=img_shape+(no_slices,), dtype='int16')
        logging.info("Merging images into 3d representation...")
        for i, slic in enumerate(tqdm(slices)):
            image_matrix[:, : , i] = plt.imread(os.path.join(path_to_image_sequence, slices[i]))
        logging.info("Done.")
        return image_matrix

