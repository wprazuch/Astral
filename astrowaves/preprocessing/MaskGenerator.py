from astrowaves.data.DataLoader import DataLoader
import astrowaves.animations.animation_tools as anim_tools
import numpy as np
from tqdm import tqdm
import cv2
import os


class MaskGenerator():

    def __init__(self):
        pass

    def run(self, waves):
        pass

    def perform_thresholding(self, waves):

        mean_pixels = np.mean(waves, axis=2)
        std_pixels = np.std(waves, axis=2)
        waves_detected = np.zeros(waves.shape, dtype='uint8')

        for i in tqdm(range(waves.shape[0])):
            for j in range(waves.shape[1]):
                slic = waves[i, j, :]
                threshold = mean_pixels[i, j] + 1.0 * std_pixels[i, j]
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


if __name__ == "__main__":

    debug_path = 'C:\\Users\\Wojtek\\Documents\\Doktorat\\AstrocyteCalciumWaveDetector\\debug'

    data_loader = DataLoader()

    waves = data_loader.load_waves()

    mask_generator = MaskGenerator()

    waves_threshold = mask_generator.perform_thresholding(waves)

    anim_tools.visualize_waves(waves_threshold, filename='waves_thresh.mp4')

    waves_morph = mask_generator.perform_morphological_operations(waves_threshold)

    anim_tools.visualize_waves(waves_morph, filename='waves_thresh_morph_std1.mp4')

    np.save(os.path.join(debug_path, "waves_morph.npy"), waves_morph)
