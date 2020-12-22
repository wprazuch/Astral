import numpy as np
from tqdm import tqdm
import cv2
import os
import argparse
from skimage import io

from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


class MaskGenerator():

    def perform_morphological_operations(self, waves):

        se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))

        for i in tqdm(range(waves.shape[2])):
            slic = waves[:, :, i]
            mask = cv2.morphologyEx(slic, cv2.MORPH_OPEN, se1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se2)
            waves[:, :, i] = mask
        return waves

    def label_waves(self, mask, is_watershed):

        if is_watershed:
            D = ndimage.distance_transform_edt(mask).astype(np.float32)
            local_max = peak_local_max(D, indices=False, min_distance=20,
                                       labels=mask)
            # perform a connected component analysis on the local peaks,
            # using 8-connectivity, then appy the Watershed algorithm
            markers = ndimage.label(local_max, structure=np.ones((3, 3, 3)))[0].astype(np.uint16)
            mask = watershed(-D, markers, mask=mask, watershed_line=True)
        else:
            mask = ndimage.label(mask)[0].astype(np.uint16)

        return mask

    def calculate_std_mean_matrices(self, waves, x_range, y_range):
        if x_range == 1 and y_range == 1:
            mean_pixels = np.mean(waves, axis=2, dtype=np.float32)
            std_pixels = np.std(waves, axis=2, dtype=np.float32)
        else:
            mean_pixels, std_pixels = np.zeros(waves.shape[:2]), np.zeros(waves.shape[:2])
            for i in range(0, waves.shape[0], y_range):
                for j in range(0, waves.shape[1], x_range):
                    subspace = waves[i:i + y_range, j:j + x_range, :].copy()

                    subspace_mean_pixels = np.mean(subspace, dtype=np.float32)
                    subspace_std_pixels = np.std(subspace, dtype=np.float32)

                    mean_pixels[i:i + y_range, j:j + x_range] = subspace_mean_pixels
                    std_pixels[i:i + y_range, j:j + x_range] = subspace_std_pixels

        return mean_pixels, std_pixels

    def run(self, waves, st_dev, x_range, y_range, is_watershed):

        mean_pixels, std_pixels = self.calculate_std_mean_matrices(waves, x_range, y_range)
        # waves_detected = np.zeros(waves.shape, dtype='uint8')

        threshold = (mean_pixels + st_dev * std_pixels).astype('uint8')
        threshold = np.expand_dims(threshold, axis=-1)

        waves[waves > threshold] = 255
        waves[waves <= threshold] = 0

        waves = self.perform_morphological_operations(waves)

        waves = waves.astype('bool')

        waves = self.label_waves(waves, is_watershed)

        return waves


def generate_black_and_white_tif(waves):
    waves = waves.astype('uint8')
    waves[waves > 0] = 255
    waves = np.swapaxes(waves, 0, -1)
    waves = np.swapaxes(waves, 1, 2)
    return waves


def parse_args():
    parser = argparse.ArgumentParser(prog='maskgenerator')
    parser.add_argument('--std', help='standard deviation for thresholding', default='1.2')
    parser.add_argument('--std_xrange', type=int, default=1, help='range of standard deviation in width')
    parser.add_argument('--std_yrange', type=int, default=1, help='range of standard deviation in height')
    parser.add_argument('--directory', help='output_directory')
    parser.add_argument('--use_watershed', type=int, default=1, help='Whether to apply watershed')
    parser.add_argument('--rootdir', type=str, default='/app/data', help='root directory of files')
    args = parser.parse_args()
    return args


def generate_masks(input_path, output_path, std=1.5, x_range=1, y_range=1, use_watershed=True):
    mask_generator = MaskGenerator()
    waves = np.load(input_path)
    labelled_waves = mask_generator.run(waves, std, x_range, y_range, use_watershed)

    np.save(output_path, labelled_waves)

    labelled_waves[labelled_waves > 0] = 1
    labelled_waves = labelled_waves.astype(bool)
    black_and_white = generate_black_and_white_tif(labelled_waves)
    parent_dir = os.path.dirname(output_path)
    io.imsave(os.path.join(parent_dir, 'black_and_white.tif'), black_and_white)


def main():
    args = parse_args()
    std = args.std
    directory = args.directory
    rootdir = args.rootdir
    path = os.path.join(rootdir, directory)
    x_range = args.std_xrange
    y_range = args.std_yrange
    is_watershed = bool(args.use_watershed)

    mask_generator = MaskGenerator()
    waves = np.load(os.path.join(path, 'waves.npy')).astype('uint8')
    waves = mask_generator.run(waves, float(std), x_range, y_range, is_watershed)

    # anim_tools.visualize_waves(waves_threshold, filename='waves_thresh.mp4')
    # np.save(os.path.join(path, 'waves.npy'), waves)
    # waves = mask_generator.perform_morphological_operations(waves)
    # anim_tools.visualize_waves(waves_morph, filename='waves_thresh_morph_std1.mp4')

    np.save(os.path.join(path, "labelled_waves.npy"), waves)
    black_and_white = generate_black_and_white_tif(waves)
    io.imsave(os.path.join(path, 'black_and_white.tif'), black_and_white)


if __name__ == '__main__':
    main()
    # debug()
