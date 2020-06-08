import numpy as np
import os
import skimage
from skimage import measure
from joblib import Parallel, delayed

import logging

logging.basicConfig(level=logging.INFO)


class CalciumWaveDetector():

    def __init__(self):
        pass

    def _indices_label(self, array, label):
        return np.argwhere(array == label)

    def run2(self, waves):
        waves[waves == 255] = 1
        waves = waves.astype('bool')
        slices = [i for i in range(waves.shape[2]) if not np.any(waves[:, :, i])]
        slic = slices[int(len(slices)/2)]
        first, second = waves[:, :, :slic], waves[:, :, slic:]
        logging.info('Labelling object pixels...')
        first = measure.label(first, connectivity=3).astype('uint16')
        second = measure.label(second, connectivity=3).astype('uint16')
        second = second + first.max() + 1
        second[second == first.max() + 1] = 0
        waves = np.concatenate([first, second], axis=2)
        logging.info('Filtering small objects under 30 pixels big...')
        uniq, counts = np.unique(waves, return_counts=True)
        labels = uniq[1:]
        counts = counts[1:]
        label_counts = list(zip(labels, counts))
        count_filtered = list(filter(lambda x: x[1] > 30, label_counts))
        labels, counts = zip(*count_filtered)
        logging.info('Grouping object pixels...')
        object_cords = Parallel(n_jobs=3, verbose=10)(delayed(self._indices_label)
                                                      (waves, label) for label in labels)
        return object_cords


def debug():
    debug_path = r'C:\Users\Wojtek\Documents\Doktorat\Astral\data\output_data'
    waves = np.load(os.path.join(debug_path, "waves_morph.npy"))
    detector = CalciumWaveDetector()
    waves_inds = detector.run2(waves)
    import pickle

    with open(os.path.join(debug_path, 'waves_inds.pck'), 'wb') as file:
        pickle.dump(waves_inds, file)


def __main__():
    debug_path = '/app/data/output_data'

    waves = np.load(os.path.join(debug_path, "waves_morph.npy"))
    detector = CalciumWaveDetector()
    waves_inds = detector.run2(waves)
    import pickle
    with open(os.path.join(debug_path, 'waves_inds.pck'), 'wb') as file:
        pickle.dump(waves_inds, file)


if __name__ == '__main__':
    __main__()
    # debug()
