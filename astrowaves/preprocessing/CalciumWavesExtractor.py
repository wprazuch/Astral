import pandas as pd
import numpy as np
import os
import logging
from tqdm import tqdm


class CalciumWavesExtractor():

    def __init__(self):
        pass

    def run(self, image_matrix):

        logging.info("Extracting calcium waves from 3d representation - might take a while...")
        background = np.mean(image_matrix, axis=2).astype('int16')
        background = background.reshape(background.shape+(1,))
        image_matrix = (image_matrix - background)
        image_matrix[image_matrix < 0] = 0
        minn = np.min(image_matrix).astype('int16')
        ptp = np.ptp(image_matrix).astype('int16')
        for i in tqdm(range(image_matrix.shape[2])):
            image_matrix[:, :, i] = 255*((image_matrix[:, :, i]-minn)/ptp)
            image_matrix = image_matrix.astype('uint8')

        return image_matrix


def __main__():
    cwe = CalciumWavesExtractor()
    timespace = np.load('/app/data/output_data/timespace.npy')
    waves = cwe.run(timespace)
    np.save('/app/data/output_data/waves.npy', waves)


def debug():
    path = r'C:\Users\Wojtek\Documents\Doktorat\Astral\data\output_data'
    cwe = CalciumWavesExtractor()
    timespace = np.load(os.path.join(path, 'timespace.npy'))
    waves = cwe.run(timespace)
    np.save(os.path.join(path, 'waves.npy'))


if __name__ == '__main__':
    __main__()
    # debug()
