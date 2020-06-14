from astrowaves.data.DataLoader import DataLoader
import astrowaves.animations.animation_tools as anim_tools
import numpy as np
from tqdm import tqdm
import cv2
import os
import argparse


class MaskGenerator():

    def __init__(self):
        pass

    def run(self, waves):
        pass

    def perform_thresholding(self, waves, st_dev):

        mean_pixels = np.mean(waves, axis=2)
        std_pixels = np.std(waves, axis=2)
        waves_detected = np.zeros(waves.shape, dtype='uint8')

        for i in tqdm(range(waves.shape[0])):
            for j in range(waves.shape[1]):
                slic = waves[i, j, :]
                threshold = mean_pixels[i, j] + st_dev * std_pixels[i, j]
                slic[slic > threshold] = 255
                slic[slic <= threshold] = 0
                waves_detected[i, j, :] = slic
        return waves_detected

    def perform_morphological_operations(self, waves):

        se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))

        waves_morph = waves.copy()

        for i in tqdm(range(waves.shape[2])):
            slic = waves_morph[:, :, i]
            mask = cv2.morphologyEx(slic, cv2.MORPH_OPEN, se1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se2)
            waves_morph[:, :, i] = mask
        return waves_morph


def __main__():
    parser = argparse.ArgumentParser(prog='maskgenerator')
    parser.add_argument('--std', help='standard deviation for thresholding')
    parser.add_argument('--directory', help='output_directory')
    parser.add_argument('--rootdir', type=str, default='app/data', help='root directory of files')
    args = parser.parse_args()
    std = args.std
    directory = args.directory
    rootdir = args.rootdir
    path = os.path.join(rootdir, directory)

    mask_generator = MaskGenerator()
    waves = np.load(os.path.join(path, 'waves.npy'))
    waves_threshold = mask_generator.perform_thresholding(waves, float(std))
    # anim_tools.visualize_waves(waves_threshold, filename='waves_thresh.mp4')
    np.save(os.path.join(path, 'waves.npy'), waves_threshold)
    waves_morph = mask_generator.perform_morphological_operations(waves_threshold)
    # anim_tools.visualize_waves(waves_morph, filename='waves_thresh_morph_std1.mp4')
    np.save(os.path.join(path, "waves_morph.npy"), waves_morph)


if __name__ == '__main__':
    __main__()
