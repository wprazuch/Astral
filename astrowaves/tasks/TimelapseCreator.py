import os
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import re
import argparse
from PIL import Image


logging.basicConfig(level=logging.INFO)


class TimelapseCreator():

    def __init__(self):
        pass

    def run(self, path_to_image=None):
        img = Image.open(path_to_image)
        no_frames = img.n_frames
        timespace = np.zeros(shape=img.size[::-1] + (no_frames,))
        for i in tqdm(range(no_frames)):
            img.seek(i)
            slic = np.array(img)
            timespace[:, :, i] = slic
        timespace = timespace.astype('uint8')
        return timespace


def main():
    parser = argparse.ArgumentParser(prog='timespacecreator')
    parser.add_argument('-f', '--filename', help='filename to add')
    parser.add_argument('--directory', help='input path where images are stored')
    parser.add_argument('--rootdir', type=str, default='/app/data', help='root directory of files')
    args = parser.parse_args()
    input_file = args.filename
    directory = args.directory
    rootdir = args.rootdir

    input_path = os.path.join(rootdir, input_file)
    output_path = os.path.join(rootdir, directory)

    ts_creator = TimelapseCreator()
    timespace = ts_creator.run(input_path).astype('uint8')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    np.save(os.path.join(output_path, 'timespace.npy'), timespace)


def debug():
    input_file = 'Cont_AN_2_4.tif'
    directory = 'Cont_AN_2_4'
    rootdir = r'C:\Users\Wojtek\Documents\Doktorat\Astral\data'

    input_path = os.path.join(rootdir, input_file)
    output_path = os.path.join(rootdir, directory)

    ts_creator = TimelapseCreator()
    timespace = ts_creator.run(input_path).astype('uint8')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    np.save(os.path.join(output_path, 'timespace.npy'), timespace)


if __name__ == '__main__':
    # debug()
    main()
