import os
import numpy as np
import pandas as pd
import cv2 as cv
import nrrd
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)


class CalciumWaveTimeSpace():

    def __init__(self):
        pass

    def create_time_space(self, path_to_image_sequence):
        slices = [slic for slic in os.listdir(path_to_image_sequence) if slic.endswith('.tif') and not slic.startswith('.')]
        no_slices = len(slices)
        logging.info("Detected {} image slices".format(no_slices))
        I = plt.imread(''.join([path_to_image_sequence, slices[no_slices-1]]))
        img_shape = I.shape
        logging.info("Detected image shape of {} pixels".format(img_shape))
        image_matrix = np.ndarray(shape=img_shape+(no_slices,), dtype='uint8')
        logging.info("Merging images into 3d representation...")
        for i, slic in enumerate(tqdm(slices)):
            image_matrix[:, : , i] = plt.imread(''.join([path_to_image_sequence, slices[i]]))
        logging.info("Done.")
        return image_matrix
        

    def create_calcium_waves_map(self, image_matrix):
        logging.info("Extracting calcium waves from 3d representation - might take a while...")
        background = np.mean(image_matrix, axis=2).astype('int16')
        background = background.reshape(background.shape+(1,))
        waves = np.ndarray(shape=image_matrix.shape, dtype='int16')
        waves = (image_matrix-background)
        waves_norm = np.ndarray(shape=image_matrix.shape, dtype='int16')
        minn = np.min(waves).astype('int16')
        ptp = np.ptp(waves).astype('int16')
        waves_norm = 255*((waves-minn)/ptp)
        waves_norm = waves_norm.astype('uint8')

        del waves, background

        return waves_norm