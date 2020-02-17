import os
import numpy as np
import pandas as pd
import cv2 as cv
import nrrd
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import re

logging.basicConfig(level=logging.INFO)


class CalciumWaveTimeSpace():

    def __init__(self):
        pass

    def create_time_space(self, path_to_image_sequence):
        slices = [slic for slic in os.listdir(path_to_image_sequence) if slic.endswith('.tif') and not slic.startswith('.')]
        slices.sort(key=lambda f: int(re.sub('\D', '', f)))
        no_slices = len(slices)
        logging.info("Detected {} image slices".format(no_slices))
        I = plt.imread(''.join([path_to_image_sequence, slices[no_slices-1]]))
        img_shape = I.shape
        logging.info("Detected image shape of {} pixels".format(img_shape))
        image_matrix = np.ndarray(shape=img_shape+(no_slices,), dtype='int16')
        logging.info("Merging images into 3d representation...")
        for i, slic in enumerate(tqdm(slices)):
            image_matrix[:, : , i] = plt.imread(''.join([path_to_image_sequence, slices[i]]))
        logging.info("Done.")
        return image_matrix


    def create_calcium_waves_map(self, image_matrix):
        logging.info("Extracting calcium waves from 3d representation - might take a while...")
        background = np.mean(image_matrix, axis=2).astype('int16')
        background = background.reshape(background.shape+(1,))
        image_matrix = (image_matrix-background)
        image_matrix[image_matrix<0] = 0
        minn = np.min(image_matrix).astype('int16')
        ptp = np.ptp(image_matrix).astype('int16')

        # for i in tqdm(range(image_matrix.shape[0])):
        #     for j in range(image_matrix.shape[1]):
        #             image_matrix[i, j, :] = 255*((image_matrix[i,j,:] - minn) / ptp)

        image_matrix = image_matrix.astype('int16')
        
        return image_matrix
