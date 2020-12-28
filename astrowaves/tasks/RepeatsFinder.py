import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import itertools
import argparse
import pandas as pd
import pickle

import logging


class RepeatsFinder():

    def _do_overlap(self, shape1, shape2, threshold):
        logging.debug('Inside _do_overlap')
        no_intersected = self._multidim_intersect(shape1, shape2)
        larger_x = sorted([shape1, shape2], key=len)[-1].shape[0]

        return no_intersected / larger_x >= threshold

    def _do_lists_intersect(self, list_a, list_b):
        return bool(set(list_a) & set(list_b))

    def _get_shape_voxels_by_id(self, abs_csv, shape_id):
        return abs_csv.loc[abs_csv['id'] == shape_id]

    def _get_z_projection(self, shape_id, abs_df):
        logging.debug('Getting z projection of shape {}'.format(shape_id))
        proj = np.unique(abs_df.loc[abs_df['id'] == shape_id, ['x', 'y']].values.astype('int16'), axis=0)
        return proj

    def _intersection_over_union(self, shape1, shape2):
        intersection = self._intersection2d(shape1, shape2)
        union = np.unique(np.vstack([shape1, shape2]), axis=0)
        return intersection.shape[0] / union.shape[0]

    def _intersection2d(self, X, Y):
        """
        Function to find intersection of two 2D arrays.
        Returns index of rows in X that are common to Y.
        """
        logging.debug('Inside _intersection2d')
        X = np.tile(X[:, :, None], (1, 1, Y.shape[0]))
        Y = np.swapaxes(Y[:, :, None], 0, 2)
        Y = np.tile(Y, (X.shape[0], 1, 1))
        eq = np.all(np.equal(X, Y), axis=1)
        eq = np.any(eq, axis=1)
        return np.nonzero(eq)[0]

    def _multidim_intersect(self, arr1, arr2):
        arr1_view = arr1.view([('', arr1.dtype)]*arr1.shape[1])
        arr2_view = arr2.view([('', arr2.dtype)]*arr2.shape[1])
        intersected = np.intersect1d(arr1_view, arr2_view)
        no_intersected = intersected.view(arr1.dtype).reshape(-1, arr1.shape[1]).shape[0]
        logging.debug(f'{no_intersected} points intersect!')
        return no_intersected

    def _remove_duplicate_lists_from_list(self, list_of_lists):
        list_of_lists.sort()
        return list(list_of_lists for list_of_lists, _ in itertools.groupby(list_of_lists))

    def _merge_repeats(self, repeats_list):
        changed = True

        while changed:
            changed = False
            for i in range(len(repeats_list)):
                repeat = repeats_list[i]
                for j in range(len(repeats_list)):
                    if i != j:
                        repeat2 = repeats_list[j]
                        if not set(repeat).isdisjoint(repeat2):
                            changed = True
                            repeats_list[i].extend(repeats_list[j])
                            del repeats_list[j]
                            break
                if changed:
                    break

        repeats_list = [sorted(list(set(repeat))) for repeat in repeats_list]
        return repeats_list

    def _exclude_repeats_from_singles(self, all_repeat_ids, singles):
        for repeat in all_repeat_ids:
            if repeat in singles:
                singles.remove(repeat)
        return singles

    def run(self, threshold, abs_df, neighbor_df):
        ids = np.unique(neighbor_df['shape_id_1'].values).tolist()

        repeats = defaultdict(list)
        singles = []

        for shape1_id in tqdm(ids):
            #     print(f"Shape1 id: {shape1_id}")
            logging.debug('Getting all neighbours of shape %s' % shape1_id)

            shape1_id_repeats = self.search_for_repeats_of(shape1_id, neighbor_df, abs_df, threshold)

            if len(shape1_id_repeats) == 0:
                singles.append(shape1_id)
            else:
                repeats[shape1_id] = shape1_id_repeats

        if repeats:
            repeats_uq = {key: value for key, value in repeats.items() if value}
            single_additional = [key for key, value in repeats_uq.items() if len(value) < 2]
            singles.extend(single_additional)
            repeats_uq = {key: value for key, value in repeats_uq.items() if len(value) >= 2}
            repeats_l = [[key, *value] for key, value in repeats_uq.items()]
            repeats_final = self._merge_repeats(repeats_l.copy())
            all_repeat_ids = [x for item in repeats_final for x in item]
            singles = self._exclude_repeats_from_singles(all_repeat_ids, singles)
            return singles, repeats_final
        else:
            return singles, []

    def search_for_repeats_of(self, shape_id, neighbors_df, absolute_df, threshold):
        neighbors = neighbors_df.loc[neighbors_df['shape_id_1'] == shape_id]['shape_id_2'].values
        repeats = []

        for shape2_id in neighbors:
            logging.debug('Getting z projections...')
            shape1_proj = self._get_z_projection(shape_id, absolute_df)
            shape2_proj = self._get_z_projection(shape2_id, absolute_df)

            logging.debug('Got projections')
            if self._do_overlap(shape1_proj, shape2_proj, threshold):
                logging.debug('Found repeat')
                repeats.append(shape2_id)
            else:
                logging.debug('Next...')

        return repeats


def find_repeats(input_path, output_path, intersect_threshold=0.8):
    abs_df = pd.read_hdf(os.path.join(input_path, 'segmentation_absolute.h5')).astype('int16')
    abs_df = abs_df[['id', 'x', 'y']]

    neighbors_df = pd.read_csv(os.path.join(input_path, 'neighbors.csv')).astype('int16')

    repeats_finder = RepeatsFinder()

    singles, repeats = repeats_finder.run(intersect_threshold, abs_df, neighbors_df)

    with open(os.path.join(output_path, 'singles.pickle'), 'wb') as handle:
        pickle.dump(singles, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(output_path, 'repeats.pickle'), 'wb') as handle:
        pickle.dump(repeats, handle, protocol=pickle.HIGHEST_PROTOCOL)


def parse_args():
    parser = argparse.ArgumentParser(prog='RepeatsFinder')
    parser.add_argument('--directory', help='output_directory')
    parser.add_argument('--rootdir', type=str, default='/app/data', help='root directory of files')
    parser.add_argument('--intersect_threshold', help='intersection threshold')
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = args.rootdir
    directory = args.directory

    intersect_threshold = float(args.intersect_threshold)
    input_path = os.path.join(root_dir, directory)

    logging.basicConfig(filename=os.path.join(input_path, 'logging.log'), level=logging.DEBUG)
    logging.info('Starting RepeatsFinder')

    abs_df = pd.read_hdf(os.path.join(input_path, 'segmentation_absolute.h5')).astype('int16')
    abs_df = abs_df[['id', 'x', 'y']]

    neighbors_df = pd.read_csv(os.path.join(input_path, 'neighbors.csv')).astype('int16')

    logging.info('Loaded metadata neighbors and absolute voxels indices.')

    repeats_finder = RepeatsFinder()

    logging.info('Starting repeats finding...')
    singles, repeats = repeats_finder.run(intersect_threshold, abs_df, neighbors_df)

    logging.info('Pickling singles and repeats...')

    with open(os.path.join(input_path, 'singles.pickle'), 'wb') as handle:
        pickle.dump(singles, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(input_path, 'repeats.pickle'), 'wb') as handle:
        pickle.dump(repeats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info('Done.')


if __name__ == '__main__':
    main()
