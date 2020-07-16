import numpy as np
import os
import skimage
from skimage import measure
from joblib import Parallel, delayed
from tqdm import tqdm
import argparse


class CalciumWaveDetector():

    def __init__(self):
        pass

    def _indices_label(self, array, label, offset):
        indices = np.argwhere(array == label)
        indices = [np.concatenate([elem[:-1], [elem[-1] + offset]]) for elem in indices]
        return indices

    def run(self, waves):
        waves_labelled = measure.label(waves, connectivity=3).astype('uint16')
        uniq, counts = np.unique(waves_labelled, return_counts=True)
        labels = uniq[1:]
        counts = counts[1:]
        label_counts = list(zip(labels, counts))
        count_filtered = list(filter(lambda x: x[1] > 30, label_counts))
        labels, counts = zip(*count_filtered)
        object_cords = Parallel(n_jobs=5, verbose=10)(delayed(self._indices_label)
                                                      (waves_labelled, label) for label in labels)
        return object_cords

    def run2(self, waves, volume_threshold):
        slices = [slic for slic in range(waves.shape[2]) if not np.any(waves[:, :, slic])]
        length = waves.shape[2]
        to_slice = [i*50 for i in range(int(length/50))]
        def func(myList, myNumber): return min(myList, key=lambda x: abs(x - myNumber))
        out = list(map(lambda x: func(slices, x), to_slice))
        out = [*out, length]
        out = sorted(list(set(out)))

        total = []

        for index in tqdm(range(len(out) - 1)):
            current = waves[:, :, out[index]:out[index + 1]]
            labelled = measure.label(current, connectivity=3).astype('uint16')
            last_slice = index
            uniq, counts = np.unique(labelled, return_counts=True)
            labels = uniq[1:]
            counts = counts[1:]
            label_counts = list(zip(labels, counts))
            count_filtered = list(filter(lambda x: x[1] > volume_threshold, label_counts))
            if not count_filtered:
                continue
            labels, counts = zip(*count_filtered)
            object_cords = Parallel(n_jobs=3, verbose=10)(delayed(self._indices_label)
                                                          (labelled, label, out[index]) for label in labels)
            total.extend(object_cords)
        return total


def parse_args():
    parser = argparse.ArgumentParser(prog='timespacecreator')
    parser.add_argument('--volume_threshold', help='standard deviation for thresholding')
    parser.add_argument('--directory', help='output_directory')
    parser.add_argument('--rootdir', type=str, default='/app/data', help='root directory of files')
    args = parser.parse_args()
    return args


def debug():
    debug_path = r'C:\Users\Wojtek\Documents\Doktorat\Astral\data\Cont_AN_2_4'
    waves = np.load(os.path.join(debug_path, "waves_morph.npy"))
    detector = CalciumWaveDetector()
    waves_inds = detector.run2(waves, 45)
    import pickle

    with open(os.path.join(debug_path, 'waves_inds.pck'), 'wb') as file:
        pickle.dump(waves_inds, file)


def main():
    args = parse_args()
    volume_threshold = args.volume_threshold
    directory = args.directory
    rootdir = args.rootdir

    path = os.path.join(rootdir, directory)

    waves = np.load(os.path.join(path, "waves_morph.npy"))
    detector = CalciumWaveDetector()
    waves_inds = detector.run2(waves, int(volume_threshold))
    import pickle

    with open(os.path.join(path, 'waves_inds.pck'), 'wb') as file:
        pickle.dump(waves_inds, file)


if __name__ == '__main__':
    # debug()
    main()
