import pandas as pd
import numpy as np
import os
import logging


class CalciumWavesExtractor():

    def __init__(self, input_file="timespace.npy", output_path=None):
        self.input_file = input_file
        self.output_path = output_path
        self.filename = 'waves.npy'

    def run(self, input_file=None, output_path=None):

        if input_file is not None:
            self.input_file = input_file
        if output_path is not None:
            self.output_path = output_path

        image_matrix=np.load(input_file)

        logging.info("Extracting calcium waves from 3d representation - might take a while...")
        background = np.mean(image_matrix, axis=2).astype('int16')
        background = background.reshape(background.shape+(1,))
        image_matrix = (image_matrix-background)
        image_matrix[image_matrix < 0] = 0
        minn = np.min(image_matrix).astype('int16')
        ptp = np.ptp(image_matrix).astype('int16')

        image_matrix = image_matrix.astype('int16')
        np.save(os.path.join(self.output_path, self.filename), image_matrix)