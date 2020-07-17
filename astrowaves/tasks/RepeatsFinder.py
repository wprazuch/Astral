import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import itertools
import argparse
import pandas as pd
import pickle


class RepeatsFinder():

    def __init__(self):
        pass

    def _intersection_over_union(self, shape1, shape2):
        intersection = self._intersection2d(shape1, shape2)
        union = np.unique(np.vstack([shape1, shape2]), axis=0)
        return intersection.shape[0] / union.shape[0]

    def _intersection2d(self, X, Y):
        """
        Function to find intersection of two 2D arrays.
        Returns index of rows in X that are common to Y.
        """
        X = np.tile(X[:, :, None], (1, 1, Y.shape[0]))
        Y = np.swapaxes(Y[:, :, None], 0, 2)
        Y = np.tile(Y, (X.shape[0], 1, 1))
        eq = np.all(np.equal(X, Y), axis=1)
        eq = np.any(eq, axis=1)
        return np.nonzero(eq)[0]

    def _get_shape_voxels_by_id(self, abs_csv, shape_id):
        return abs_csv.loc[abs_csv['id'] == shape_id]

    def _get_z_projection(self, shape):
        shape = shape[['x', 'y']]
        shape = np.unique(shape, axis=0)
        return shape

    def _do_overlap(self, shape1, shape2, threshold):
        intersected = self._intersection2d(shape1, shape2)
        iou = self._intersection_over_union(shape1, shape2)
        larger = sorted([shape1, shape2], key=len)[-1]

        return intersected.shape[0] / larger.shape[0] >= threshold

    def _do_lists_intersect(self, list_a, list_b):
        return bool(set(list_a) & set(list_b))

    def _remove_duplicate_lists_from_list(self, list_of_lists):
        list_of_lists.sort()
        return list(list_of_lists for list_of_lists, _ in itertools.groupby(list_of_lists))

    def _merge_repeats(self, repeats_list):
        final_repeats = []
        for i in range(len(repeats_list)):
            new_repeats = repeats_list[i]
            for j in range(i + 1, len(repeats_list)):
                if i != j:
                    if self._do_lists_intersect(repeats_list[i], repeats_list[j]):
                        new_repeats.extend(repeats_list[j])
            new_repeats = list(set(new_repeats))
            final_repeats.append(new_repeats)

        final_repeats = self._remove_duplicate_lists_from_list(final_repeats)
        return final_repeats

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
            shape1 = self._get_shape_voxels_by_id(abs_df, shape1_id)
        #     print(f"Shape1 id: {shape1_id}")
            neighbors = neighbor_df.loc[neighbor_df['shape_id_1'] == shape1_id]['shape_id_2'].values
        #     print(f"Neighbors : {neighbors}")
            for shape2_id in neighbors:
                shape2 = self._get_shape_voxels_by_id(abs_df, shape2_id)
        #         print(f"Shape2 id: {shape2_id}")
                shape1_proj = self._get_z_projection(shape1)
                shape2_proj = self._get_z_projection(shape2)
                if self._do_overlap(shape1_proj, shape2_proj, threshold):
                    repeats[shape1_id].append(shape2_id)
            if not repeats[shape1_id]:
                singles.append(shape1_id)

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


def parse_args():
    parser = argparse.ArgumentParser(prog='RepeatsFinder')
    parser.add_argument('--directory', help='output_directory')
    parser.add_argument('--rootdir', type=str, default='/app/data', help='root directory of files')
    parser.add_argument('--intersect_threshold', help='intersection threshold')
    return parser.parse_args()


def debug():
    args = parse_args()
    root_dir = r'C:\Users\Wojtek\Documents\Doktorat\Astral\data'
    directory = 'Cont_AA_1_2'
    directory_path = os.path.join(root_dir, directory)
    abs_df = pd.read_hdf(os.path.join(directory_path, 'segmentation_absolute.h5'))
    neighbors_df = pd.read_csv(os.path.join(directory_path, 'neighbors.csv'))
    repeats_finder = RepeatsFinder()
    singles, repeats = repeats_finder.run(0.8, abs_df, neighbors_df)

    with open(r'C:\Users\Wojtek\Documents\Doktorat\Astral\data\singles.pickle', 'wb') as handle:
        pickle.dump(singles, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(r'C:\Users\Wojtek\Documents\Doktorat\Astral\data\repeats.pickle', 'wb') as handle:
        pickle.dump(repeats, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    args = parse_args()
    root_dir = args.rootdir
    directory = args.directory
    intersect_threshold = float(args.intersect_threshold)
    directory_path = os.path.join(root_dir, directory)
    abs_df = pd.read_hdf(os.path.join(directory_path, 'segmentation_absolute.h5'))
    neighbors_df = pd.read_csv(os.path.join(directory_path, 'neighbors.csv'))
    repeats_finder = RepeatsFinder()
    singles, repeats = repeats_finder.run(intersect_threshold, abs_df, neighbors_df)

    with open(os.path.join(directory_path, 'singles.pickle'), 'wb') as handle:
        pickle.dump(singles, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(directory_path, 'repeats.pickle'), 'wb') as handle:
        pickle.dump(repeats, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
