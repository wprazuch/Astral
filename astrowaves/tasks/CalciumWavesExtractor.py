import pandas as pd
import numpy as np
import os
import logging
from tqdm import tqdm
import argparse
import math


class CalciumWavesExtractor():

    def __init__(self):
        pass

    def _extract_background(self, timelapse):
        background = np.mean(timelapse, axis=2).astype('int16')
        background = background.reshape(background.shape + (1,))
        timelapse = (timelapse - background)
        timelapse[timelapse < 0] = 0
        return timelapse

    def run(self, timelapse):
        logging.info("Extracting calcium waves from 3d representation - might take a while...")
        timelapse = self._extract_background(timelapse)
        minn = np.min(timelapse).astype('uint16')
        ptp = np.ptp(timelapse).astype('uint16')
        chunksize = 250
        iters = int(math.ceil(timelapse.shape[2] / chunksize))
        for i in range(iters):
            # print(i)
            # (255 * ((image_matrix - minn) / ptp)).astype('uint8')
            chunk = timelapse[..., i * chunksize: (i + 1) * chunksize]
            chunk = (255 * ((chunk - minn) / ptp)).astype('uint8')
            timelapse = timelapse.astype('uint8')

        return timelapse


def main():

    parser = argparse.ArgumentParser(prog='timespacecreator')
    parser.add_argument('--directory', help='input path where images are stored')
    parser.add_argument('--rootdir', type=str, default='/app/data', help='root directory of files')
    args = parser.parse_args()
    rootdir = args.rootdir
    directory = args.directory

    input_path = os.path.join(rootdir, directory)

    cwe = CalciumWavesExtractor()
    timespace = np.load(os.path.join(input_path, 'timespace.npy'))
    waves = cwe.run(timespace)
    np.save(os.path.join(input_path, 'waves.npy'), waves)


def debug():

    parser = argparse.ArgumentParser(prog='timespacecreator')
    parser.add_argument('--directory', help='input path where images are stored')
    parser.add_argument('--rootdir', type=str, default='/app/data', help='root directory of files')
    args = parser.parse_args()
    rootdir = r'C:\\Users\Wojtek\Documents\Doktorat\Astral\data'
    directory = 'Cont_AN_2_2'

    input_path = os.path.join(rootdir, directory)

    cwe = CalciumWavesExtractor()
    timespace = np.load(os.path.join(input_path, 'timespace.npy'))
    waves = cwe.run(timespace)
    np.save(os.path.join(input_path, 'waves.npy'), waves)


if __name__ == '__main__':
    main()
    # debug()
