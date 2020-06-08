import numpy as np
import os
import skimage
from skimage import measure
from joblib import Parallel, delayed


class CalciumWaveDetector():

    def __init__(self):
        pass

    def _indices_label(self, array, label):
        return np.argwhere(array == label)

    def run2(self, waves):
        waves_labelled = measure.label(waves, connectivity=3).astype('uint16')
        uniq, counts = np.unique(waves_labelled, return_counts=True)
        labels = uniq[1:]
        counts = counts[1:]
        label_counts = list(zip(labels, counts))
        count_filtered = list(filter(lambda x: x[1] > 30, label_counts))
        labels, counts = zip(*count_filtered)
        object_cords = Parallel(n_jobs=3, verbose=10)(delayed(self._indices_label)
                                                      (waves_labelled, label) for label in labels)
        return object_cords


if __name__ == '__main__':

    debug_path = '/app/data/output_data'

    waves = np.load(os.path.join(debug_path, "waves_morph.npy"))

    detector = CalciumWaveDetector()

    #wave_inds = np.argwhere(waves == 255)

    waves_inds = detector.run2(waves)

    import pickle

    with open(os.path.join(debug_path, 'waves_inds.pck'), 'wb') as file:
        pickle.dump(waves_inds, file)

    #seg = region_grow(waves, wave_inds[2])
