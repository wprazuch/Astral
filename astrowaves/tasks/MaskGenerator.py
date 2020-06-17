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

        mean_pixels = np.mean(waves, axis=2, dtype=np.float32)
        std_pixels = np.std(waves, axis=2, dtype=np.float32)
        #waves_detected = np.zeros(waves.shape, dtype='uint8')

        for i in tqdm(range(waves.shape[0])):
            for j in range(waves.shape[1]):
                slic = waves[i, j, :]
                threshold = int(mean_pixels[i, j] + st_dev * std_pixels[i, j])
                slic[slic > threshold] = 255
                slic[slic <= threshold] = 0

        print(np.unique(waves))
        se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))

        for i in tqdm(range(waves.shape[2])):
            slic = waves[:, :, i]
            mask = cv2.morphologyEx(slic, cv2.MORPH_OPEN, se1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se2)
            waves[:, :, i] = mask

        waves = waves.astype('bool')
        print(np.unique(waves))
        return waves

    def perform_morphological_operations(self, waves_morph):

        se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))

        for i in tqdm(range(waves_morph.shape[2])):
            slic = waves_morph[:, :, i]
            mask = cv2.morphologyEx(slic, cv2.MORPH_OPEN, se1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se2)
            waves_morph[:, :, i] = mask
        return waves_morph


def parse_args():
    parser = argparse.ArgumentParser(prog='maskgenerator')
    parser.add_argument('--std', help='standard deviation for thresholding', default='1.2')
    parser.add_argument('--directory', help='output_directory')
    parser.add_argument('--rootdir', type=str, default='/app/data', help='root directory of files')
    args = parser.parse_args()
    return args


def debug():
    directory = 'Cont_AN_2_4'
    rootdir = r'C:\Users\Wojtek\Documents\Doktorat\Astral\data'
    path = os.path.join(rootdir, directory)
    std = 1.2

    mask_generator = MaskGenerator()
    waves = np.load(os.path.join(path, 'waves.npy')).astype('uint8')
    waves = mask_generator.perform_thresholding(waves, float(std))
    # anim_tools.visualize_waves(waves_threshold, filename='waves_thresh.mp4')
    # np.save(os.path.join(path, 'waves.npy'), waves)
    # waves = mask_generator.perform_morphological_operations(waves)
    # anim_tools.visualize_waves(waves_morph, filename='waves_thresh_morph_std1.mp4')
    np.save(os.path.join(path, "waves_morph.npy"), waves)


def __main__():
    args = parse_args()
    std = args.std
    directory = args.directory
    rootdir = args.rootdir
    path = os.path.join(rootdir, directory)

    mask_generator = MaskGenerator()
    waves = np.load(os.path.join(path, 'waves.npy')).astype('uint8')
    waves = mask_generator.perform_thresholding(waves, float(std))
    # anim_tools.visualize_waves(waves_threshold, filename='waves_thresh.mp4')
    # np.save(os.path.join(path, 'waves.npy'), waves)
    # waves = mask_generator.perform_morphological_operations(waves)
    # anim_tools.visualize_waves(waves_morph, filename='waves_thresh_morph_std1.mp4')
    np.save(os.path.join(path, "waves_morph.npy"), waves)


if __name__ == '__main__':
    __main__()
    # debug()
