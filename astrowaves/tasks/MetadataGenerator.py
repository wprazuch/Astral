import numpy as np
import pandas as pd
import os
import pickle
import argparse
from .. import config
from .models.rows import DimensionsRow


class MetadataGenerator():

    def run(self, waves_ind_list, timespace):
        data_relative = np.ndarray(shape=(0, 5))
        data_absolute = np.ndarray(shape=(0, 5), dtype='int32')

        abs_df_col_names = list(config.ABS_DF_COLUMNS.keys())
        rel_df_col_names = list(config.REL_DF_COLUMNS.keys())
        dims_df_col_names = list(config.DIMENSIONS_DF_COLUMNS.keys())

        indices_sorted = list(reversed(sorted(waves_ind_list.copy(), key=len)))
        indices_filtered_volume = list(filter(lambda x: len(x) > 200, indices_sorted))

        data_dims = np.zeros(shape=(len(indices_filtered_volume), 10), dtype='int32')

        for i, indices in enumerate(indices_filtered_volume):
            max_x, min_x, max_y, min_y, max_z, min_z = self.get_extrema_cords(indices)
            center_x, center_y, center_z = self.get_center_cords(max_x, min_x, max_y, min_y, max_z, min_z)

            roi_shape, indices_shift = self.get_index_shift(indices, max_x, min_x, max_y, min_y, max_z, min_z)

            roi = np.zeros(shape=roi_shape, dtype='uint16')

            color = np.ones((indices_shift.shape[0], 1), dtype='uint16')

            dims_row = self.get_dims_data_row(i, max_x, min_x, max_y, min_y,
                                              max_z, min_z, center_x, center_y, center_z)

            data_dims[i, :] = dims_row

            for index, ref_index in zip(indices_shift, indices):
                x, y, z = index
                x_ref, y_ref, z_ref = ref_index
                roi[x, y, z] = timespace[x_ref, y_ref, z_ref]

            for j, index in enumerate(indices_shift):
                x, y, z = index
                color[j] = roi[x, y, z]

            id_ = i * np.ones((indices_shift.shape[0], 1), dtype='uint16')

            data_r = np.concatenate([id_, indices_shift, color], axis=1)
            data_a = np.concatenate([id_, indices, color], axis=1)
            data_relative = np.concatenate([data_relative, data_r], axis=0)
            data_relative = data_relative.astype('uint16')
            data_absolute = np.concatenate([data_absolute, data_a], axis=0)
            data_absolute = data_absolute.astype('int32')

        rel_df = pd.DataFrame(columns=rel_df_col_names, data=data_relative)
        abs_df = pd.DataFrame(columns=abs_df_col_names, data=data_absolute)
        dims_df = pd.DataFrame(columns=dims_df_col_names, data=data_dims)

        return (abs_df, rel_df, dims_df)

    def get_dims_data_row(self, i, max_x, min_x, max_y, min_y, max_z, min_z, center_x, center_y, center_z):
        row_obj = DimensionsRow(i, min_y, max_y, min_x, max_x, min_z, max_z, center_y, center_x, center_z)
        row = row_obj.get_row()
        return row

    def get_index_shift(self, indices, max_x, min_x, max_y, min_y, max_z, min_z):
        min_ind, max_ind = np.array((min_x, min_y, min_z)), np.array((max_x, max_y, max_z))
        roi_shape = max_ind - min_ind + [1, 1, 1]
        indices_shift = indices.copy()
        indices_shift = indices_shift - min_ind
        return roi_shape, indices_shift

    def get_center_cords(self, max_x, min_x, max_y, min_y, max_z, min_z):
        center_x = np.mean([max_x, min_x])
        center_y = np.mean([max_y, min_y])
        center_z = np.mean([max_z, min_z])
        return center_x, center_y, center_z

    def get_extrema_cords(self, indices):
        max_x = max(indices, key=lambda x: x[0])[0]
        min_x = min(indices, key=lambda x: x[0])[0]
        max_y = max(indices, key=lambda x: x[1])[1]
        min_y = min(indices, key=lambda x: x[1])[1]
        max_z = max(indices, key=lambda x: x[2])[2]
        min_z = min(indices, key=lambda x: x[2])[2]
        return max_x, min_x, max_y, min_y, max_z, min_z


def generate_metadata(input_path, output_path):

    timespace = np.load(os.path.join(input_path, 'timespace.npy'))

    with open(os.path.join(input_path, 'waves_inds.pck'), 'rb') as f:
        waves_inds = pickle.load(f)

    metadata_generator = MetadataGenerator()

    abs_df, rel_df, dims_df = metadata_generator.run(waves_inds, timespace)

    rel_df.to_hdf(os.path.join(output_path, 'segmentation_relative.h5'), key='df')
    abs_df.to_hdf(os.path.join(output_path, 'segmentation_absolute.h5'), key='df')
    dims_df.to_hdf(os.path.join(output_path, 'segmentation_dims.h5'), key='df')


def main():
    parser = argparse.ArgumentParser(prog='Segmenter')
    parser.add_argument('--directory', help='output_directory')
    parser.add_argument('--rootdir', type=str, default='/app/data', help='root directory of files')

    args = parser.parse_args()
    directory = args.directory
    rootdir = args.rootdir

    path = os.path.join(rootdir, directory)
    timespace = np.load(os.path.join(path, 'timespace.npy'))

    with open(os.path.join(path, 'waves_inds.pck'), 'rb') as f:
        waves_inds = pickle.load(f)

    metadata_generator = MetadataGenerator()

    abs_df, rel_df, dims_df = metadata_generator.run(waves_inds, timespace)

    rel_df.to_hdf(os.path.join(path, 'segmentation_relative.h5'), key='df')
    abs_df.to_hdf(os.path.join(path, 'segmentation_absolute.h5'), key='df')
    dims_df.to_hdf(os.path.join(path, 'segmentation_dims.h5'), key='df')

    print('Done')


if __name__ == '__main__':

    main()
