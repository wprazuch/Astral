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

    def run(self, waves):

        # get pixels of class 'wave'
        wave_inds = np.argwhere(waves == 255)
        calcium_waves = []

        while len(wave_inds) > 0:
            calcium_wave = self.region_grow(waves, wave_inds[0])
            calcium_waves.append(calcium_wave)
            wave_inds = np.argwhere(waves == 255)

        return calcium_waves

    def region_grow(self, vol, start_point):

        sizez = vol.shape[0] - 1
        sizex = vol.shape[1] - 1
        sizey = vol.shape[2] - 1

        items = []
        visited = []

        def enqueue(item):
            items.insert(0, item)

        def dequeue():
            s = items.pop()
            visited.append(s)
            return s

        enqueue((start_point[0], start_point[1], start_point[2]))

        while not items == []:

            z, x, y = dequeue()
            voxel = vol[z, x, y]
            vol[z, x, y] = 1

            if x < sizex:
                tvoxel = vol[z, x+1, y]
                if tvoxel == 255:
                    enqueue((z, x+1, y))
                    vol[z, x+1, y] = 1

            if x > 0:
                tvoxel = vol[z, x-1, y]
                if tvoxel == 255:
                    enqueue((z, x-1, y))
                    vol[z, x-1, y] = 1

            if y < sizey:
                tvoxel = vol[z, x, y+1]
                if tvoxel == 255:
                    enqueue((z, x, y+1))
                    vol[z, x, y+1] = 1

            if y > 0:
                tvoxel = vol[z, x, y-1]
                if tvoxel == 255:
                    enqueue((z, x, y-1))
                    vol[z, x, y-1] = 1

            if z < sizez:
                tvoxel = vol[z+1, x, y]
                if tvoxel == 255:
                    enqueue((z+1, x, y))
                    vol[z+1, x, y] = 1

            if z > 0:
                tvoxel = vol[z-1, x, y]
                if tvoxel == 255:
                    enqueue((z-1, x, y))
                    vol[z-1, x, y] = 1

        return visited


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
