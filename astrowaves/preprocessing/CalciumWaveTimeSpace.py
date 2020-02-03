import os
import numpy as np
import pandas as pd
import cv2 as cv
import nrrd
import matplotlib.pyplot as plt
from tqdm import tqdm



class CalciumWaveTimeSpace():

    def __init__(self):
        pass

    def create_time_space(self, path_to_image_sequence):
        slices = [slic for slic in os.listdir(path_to_image_sequence) if slic.endswith('.tif') and not slic.startswith('.')]
        print(slices)
        I = plt.imread(''.join([path_to_image_sequence, slices[1199]]))
        image_matrix = np.ndarray(shape=I.shape+(len(slices),), dtype='uint8')
        print(slices[-5:])
        for i, slic in enumerate(tqdm(slices)):
            image_matrix[:, : , i] = plt.imread(''.join([path_to_image_sequence, slices[i]]))

        return image_matrix
        

    def create_calcium_waves_map(self, image_matrix):
        background = np.mean(image_matrix, axis=2).astype('int16')
        background = background.reshape(background.shape+(1,))
        waves = np.ndarray(shape=image_matrix.shape, dtype='int16')
        waves = (image_matrix-background)
        waves_norm = np.ndarray(shape=image_matrix.shape, dtype='int16')
        minn = np.min(waves).astype('int16')
        ptp = np.ptp(waves).astype('int16')
        waves_norm = 255*((waves-minn)/ptp)
        waves_norm = waves_norm.astype('uint8')

        return waves_norm