import pandas as pd
import numpy as np
import os
import logging


class CalciumWavesExtractor():

    def __init__(self):
        pass

    def run(self, image_matrix):

        logging.info("Extracting calcium waves from 3d representation - might take a while...")
        background = np.mean(image_matrix, axis=2).astype('int16')
        background = background.reshape(background.shape+(1,))
        image_matrix = (image_matrix-background)
        image_matrix[image_matrix < 0] = 0
        waves_norm = np.ndarray(shape=image_matrix.shape, dtype='int16')
        minn = np.min(image_matrix).astype('int16')
        ptp = np.ptp(image_matrix).astype('int16')
        waves_norm = 255*((image_matrix-minn)/ptp)
        waves_norm = waves_norm.astype('uint8')

        return waves_norm