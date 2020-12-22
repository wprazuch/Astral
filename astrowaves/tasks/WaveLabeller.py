import numpy as np
from skimage.measure import label
from tqdm import tqdm
from collections import defaultdict
import argparse
import os
import pickle
import logging

from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


class WaveLabeller():

    def run(self, labelled_masks, volume_threshold):

        calcium_wave_idx_dict = defaultdict(list)

        for t_frame in tqdm(range(labelled_masks.shape[2])):

            frame = labelled_masks[:, :, t_frame].copy()

            waves_ids_present = sorted(np.unique(frame))
            if waves_ids_present[0] == 0:
                waves_ids_present = waves_ids_present[1:]

            for wave_id in waves_ids_present:
                wave_idx = np.argwhere(frame == wave_id).tolist()
                wave_idx = [[*item, t_frame] for item in wave_idx]
                calcium_wave_idx_dict[wave_id].extend(wave_idx)

        calcium_wave_idx_dict_filtered = {key: value for key,
                                          value in calcium_wave_idx_dict.items() if len(value) > volume_threshold}

        calcium_wave_idx_dict_filtered_list = [indices for indices in calcium_wave_idx_dict_filtered.values()]

        return calcium_wave_idx_dict_filtered_list


def parse_args():
    parser = argparse.ArgumentParser(prog='timespacecreator')
    parser.add_argument('--volume_threshold', help='standard deviation for thresholding')
    parser.add_argument('--directory', help='output_directory')
    parser.add_argument('--rootdir', type=str, default='/app/data', help='root directory of files')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    volume_threshold = args.volume_threshold
    directory = args.directory
    rootdir = args.rootdir

    path = os.path.join(rootdir, directory)

    logging.basicConfig(filename=os.path.join(path, 'logging.log'), level=logging.DEBUG)
    logging.info('Starting WaveLabeller')

    waves = np.load(os.path.join(path, "labelled_waves.npy"))
    detector = WaveLabeller()
    waves_inds = detector.run(waves, int(volume_threshold))
    import pickle

    with open(os.path.join(path, 'waves_inds.pck'), 'wb') as file:
        pickle.dump(waves_inds, file)


if __name__ == '__main__':
    # debug()
    main()


def label_waves(waves_mask_path, output_idx_list_path, volume_threshold):
    wave_labeller = WaveLabeller()
    wave_mask = np.load(waves_mask_path)

    output_list = wave_labeller.run(wave_mask, volume_threshold)

    with open(output_idx_list_path, 'wb') as file:
        pickle.dump(output_list, file)
