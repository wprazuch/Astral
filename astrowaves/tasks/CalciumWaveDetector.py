import numpy as np
import os
import skimage
from skimage import measure
from joblib import Parallel, delayed
from tqdm import tqdm
import argparse
import logging


class CalciumWaveDetector():

    def __init__(self):
        pass

    def _indices_label(self, array, label, offset):
        indices = np.argwhere(array == label)
        indices = [np.concatenate([elem[:-1], [elem[-1] + offset]]).tolist() for elem in indices]
        return indices

    def find_closest_slice(self, myList, myNumber):
        try:
            slic = min(myList, key=lambda x: abs(x - myNumber))
        except:
            slic = 0
        return slic

    def _find_slice_points(self, waves, axis=-1):
        if axis == -1 or axis == 2:
            slices = [slic for slic in range(waves.shape[axis]) if not np.any(waves[:, :, slic])]
        elif axis == 1:
            slices = [slic for slic in range(waves.shape[axis]) if not np.any(waves[:, slic, :])]
        else:
            slices = [slic for slic in range(waves.shape[axis]) if not np.any(waves[slic, :, :])]

        length = waves.shape[axis]

        to_slice = [i * 25 for i in range(int(length / 25))]

        out = list(map(lambda x: self.find_closest_slice(slices, x), to_slice))
        out = sorted(list(set(out)))
        out = [*out, length]
        out = sorted(list(set(out)))

        return out

    def run(self, waves, volume_threshold):
        logging.debug('Running run method')

        out = self._find_slice_points(waves, axis=-1)

        total = []

        for index in tqdm(range(len(out) - 1)):
            logging.debug(f'Starting {index} iteration')
            current = waves[:, :, out[index]:out[index + 1]]
            logging.debug(f'Labelling objects in a subspace...')
            labelled = measure.label(current, connectivity=3).astype('uint16')
            logging.debug(f'Finished labelling!')
            last_slice = index
            uniq, counts = np.unique(labelled, return_counts=True)
            labels = uniq[1:]
            counts = counts[1:]
            logging.debug(f'Got {len(labels)} labels')
            label_counts = list(zip(labels, counts))
            count_filtered = list(filter(lambda x: x[1] > volume_threshold, label_counts))
            if not count_filtered:
                continue
            labels, counts = zip(*count_filtered)
            object_cords = Parallel(n_jobs=3, verbose=10)(delayed(self._indices_label)
                                                          (labelled, label, out[index]) for label in labels)

            logging.debug(f'Finishing {index} iteration')

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
    waves_inds = detector.run(waves, 45)
    import pickle

    with open(os.path.join(debug_path, 'waves_inds.pck'), 'wb') as file:
        pickle.dump(waves_inds, file)


def main():
    args = parse_args()
    volume_threshold = args.volume_threshold
    directory = args.directory
    rootdir = args.rootdir

    path = os.path.join(rootdir, directory)

    logging.basicConfig(filename=os.path.join(path, 'logging.log'), level=logging.DEBUG)
    logging.info('Starting CalciumWaveDetector')

    waves = np.load(os.path.join(path, "waves_morph.npy"))
    detector = CalciumWaveDetector()
    waves_inds = detector.run(waves, int(volume_threshold))
    import pickle

    with open(os.path.join(path, 'waves_inds.pck'), 'wb') as file:
        pickle.dump(waves_inds, file)


if __name__ == '__main__':
    # debug()
    main()
