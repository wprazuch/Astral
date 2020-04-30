import numpy as np
import pandas as pd
import os
import pickle


class CalciumWaveSegmenter():

    def __init__(self):
        pass

    def run(self, waves_ind_list, timespace):
        data_relative = np.ndarray(shape=(0, 5))
        data_absolute = np.ndarray(shape=(0, 5), dtype='int32')

        indices_sorted = list(reversed(sorted(waves_inds.copy(), key=len)))
        indices_filtered_volume = list(filter(lambda x: len(x) > 200, indices_sorted))

        data_dims = np.zeros(shape=(len(indices_filtered_volume), 7), dtype='int32')

        for i, indices in enumerate(indices_filtered_volume):
            max_x = max(indices, key=lambda x: x[0])[0]
            min_x = min(indices, key=lambda x: x[0])[0]
            max_y = max(indices, key=lambda x: x[1])[1]
            min_y = min(indices, key=lambda x: x[1])[1]
            max_z = max(indices, key=lambda x: x[2])[2]
            min_z = min(indices, key=lambda x: x[2])[2]

            min_ind, max_ind = np.array((min_x, min_y, min_z)), np.array((max_x, max_y, max_z))
            roi_shape = max_ind - min_ind + [1, 1, 1]

            indices_shift = indices.copy()
            indices_shift = indices_shift - min_ind

            roi = np.zeros(shape=roi_shape, dtype='uint8')

            color = np.ones((indices_shift.shape[0], 1), dtype='uint8')

            dim_row = [i, min_x, max_x, min_y, max_y, min_z, max_z]
            data_dims[i, 0] = i
            data_dims[i, 1] = min_x
            data_dims[i, 2] = max_x
            data_dims[i, 3] = min_y
            data_dims[i, 4] = max_y
            data_dims[i, 5] = min_z
            data_dims[i, 6] = max_z

            for index, ref_index in zip(indices_shift, indices):
                x, y, z = index
                x_ref, y_ref, z_ref = ref_index
                roi[x, y, z] = timespace[x_ref, y_ref, z_ref]

            for j, index in enumerate(indices_shift):
                x, y, z = index
                color[j] = roi[x, y, z]

            id = i * np.ones((indices_shift.shape[0], 1), dtype='uint8')

            data_r = np.concatenate([id, indices_shift, color], axis=1)
            data_a = np.concatenate([id, indices, color], axis=1)
            data_relative = np.concatenate([data_relative, data_r], axis=0)
            data_relative = data_relative.astype('uint8')
            data_absolute = np.concatenate([data_absolute, data_a], axis=0)
            data_absolute = data_absolute.astype('int32')

        abs_cols = ['id', 'y', 'x', 'z', 'color']
        dims_cols = ['id', 'y_min', 'y_max', 'x_min', 'x_max', 'z_min', 'z_max']

        rel = pd.DataFrame(columns=abs_cols, data=data_relative)
        abs = pd.DataFrame(columns=abs_cols, data=data_absolute)
        dims = pd.DataFrame(columns=dims_cols, data=data_dims)

        return (abs, rel, dims)


if __name__ == '__main__':

    debug_path = 'C:\\Users\\Wojtek\\Documents\\Doktorat\\AstrocyteCalciumWaveDetector\\debug'

    timespace = np.load(os.path.join(debug_path, 'waves.npy'))

    with open('debug\\waves_inds.pck', 'rb') as f:
        waves_inds = pickle.load(f)

    segmenter = CalciumWaveSegmenter()

    abs, rel, dims = segmenter.run(waves_inds, timespace)

    rel.to_hdf('debug\\segmentation_relative.h5', key='df')
    abs.to_hdf('debug\\segmentation_absolute.h5', key='df')
    dims.to_hdf('debug\\segmentation_dims.h5', key='df')

    print('Done')
