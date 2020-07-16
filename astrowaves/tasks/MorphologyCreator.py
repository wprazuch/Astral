import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import itertools
import argparse
import pandas as pd
import pickle
from radiomics.shape import RadiomicsShape
import SimpleITK as sitk


class MorphologyCreator():

    def __init__(self):
        pass

    def get_shape_voxels_by_id(self, abs_csv, shape_id):
        return abs_csv.loc[abs_csv['id'] == shape_id]

    def get_shape_bbox(self, abs_csv, waves, shape_id):
        abs_shape = abs_csv.loc[abs_csv['id'] == shape_id]
        min_x, min_y, min_z = abs_shape['x'].min(), abs_shape['y'].min(), abs_shape['z'].min()
        max_x, max_y, max_z = abs_shape['x'].max(), abs_shape['y'].max(), abs_shape['z'].max()
        segmentation = waves[min_y:max_y, min_x:max_x, min_z:max_z]
        return segmentation.astype('uint8')

    def calculate_morphology(self, shape, segmentation):
        max_y = np.sum(segmentation, axis=0).max()
        max_x = np.sum(segmentation, axis=1).max()
        max_z = np.sum(segmentation, axis=2).max()

        shape[shape == 1] = 255
        sitk_img = sitk.GetImageFromArray(shape)
        sitk_mask = sitk.GetImageFromArray(segmentation)
        rs = RadiomicsShape(sitk_img, sitk_mask)
        sphericity = rs.getSphericityFeatureValue()
        sphericity = round(sphericity, 2)
        morph_dict = {
            'max_x_size': max_x,
            'max_y_size': max_y,
            'max_z_size': max_z,
            'sphericity': sphericity
        }

        return morph_dict

    def calculate_morphology_for_singles(self, singles, abs_csv, waves):
        morphology_df = pd.DataFrame(columns=['shape_id', 'max_x_size', 'max_y_size', 'max_z_size', 'sphericity'])
        for single_id in singles:
            seg = self.get_shape_bbox(abs_csv, waves, single_id)
            morph_dict = self.calculate_morphology(seg.copy(), seg)
            morph_dict['shape_id'] = int(single_id)
            morphology_df = morphology_df.append(morph_dict, ignore_index=True)
        return morphology_df

    def calculate_morphology_for_repeats(self, repeats, abs_csv, waves):
        repeat_df = pd.DataFrame(
            columns=['shape_ids', 'number_of_repeats', 'avg_sphericity', 'avg_maximum_x',
                     'avg_maximum_y', 'avg_maximum_z', 'median_inter_repeat_min_z_dist',
                     'median_inter_repeat_center_dist'])

        for repeat_series in repeats:

            z_tuples = []
            sphericities = []
            max_x_sizes = []
            max_y_sizes = []
            max_z_sizes = []

            for rep_id in repeat_series:
                shape = self.get_shape_voxels_by_id(abs_csv, rep_id)
                min_z, max_z = shape['z'].min(), shape['z'].max()
                z_tuples.append((min_z, max_z))
                seg = self.get_shape_bbox(abs_csv, waves, rep_id)
                m_dict = self.calculate_morphology(seg.copy(), seg)
                sphericities.append(m_dict['sphericity'])
                max_x_sizes.append(m_dict['max_x_size'])
                max_y_sizes.append(m_dict['max_y_size'])
                max_z_sizes.append(m_dict['max_z_size'])

            z_tuples = sorted(z_tuples, key=lambda x: x[0])
            dists = [abs(z_tuples[i + 1][0] - z_tuples[i][1]) for i in range(0, len(z_tuples) - 1)]

            centers = sorted([np.mean(z_extrema) for z_extrema in z_tuples])
            center_dists = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]

            shape_ids = str(sorted(repeat_series))[1:-1].replace(',', '_')

            no_repeats = len(repeat_series)
            mean_sphericity = round(np.mean(sphericities), 2)
            mean_max_x_size = round(np.mean(max_x_sizes), 2)
            mean_max_y_size = round(np.mean(max_y_sizes), 2)
            mean_max_z_size = round(np.mean(max_z_sizes), 2)
            med_dist = np.median(dists)
            med_centers = np.median(center_dists)

            morph_dict = {
                'shape_ids': shape_ids,
                'number_of_repeats': no_repeats,
                'avg_sphericity': mean_sphericity,
                'avg_maximum_x': mean_max_x_size,
                'avg_maximum_y': mean_max_y_size,
                'avg_maximum_z': mean_max_z_size,
                'median_inter_repeat_min_z_dist': med_dist,
                'median_inter_repeat_center_dist': med_centers
            }

            repeat_df = repeat_df.append(morph_dict, ignore_index=True)
        return repeat_df

    def run(self, singles, repeats, abs_csv, neigh_csv, waves):
        repeat_df = self.calculate_morphology_for_repeats(repeats, abs_csv, waves)
        single_df = self.calculate_morphology_for_singles(singles, abs_csv, waves)

        ids_to_delete = [item[1:] for item in repeats]
        ids_to_delete = [item for reps in ids_to_delete for item in reps]
        neigh_csv = neigh_csv.loc[(~neigh_csv['shape_id_1'].isin(ids_to_delete))]
        neigh_csv = neigh_csv.loc[(~neigh_csv['shape_id_2'].isin(ids_to_delete))]

        return single_df, repeat_df, neigh_csv


def parse_args():
    parser = argparse.ArgumentParser(prog='Segmenter')
    parser.add_argument('--directory', help='output_directory')
    parser.add_argument('--rootdir', type=str, default='/app/data', help='root directory of files')
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = args.rootdir
    directory = args.directory
    directory_path = os.path.join(root_dir, directory)

    abs_df = pd.read_hdf(os.path.join(directory_path, 'segmentation_absolute.h5'))
    waves = np.load(os.path.join(directory_path, 'waves_morph.npy'))
    neighbors_df = pd.read_csv(os.path.join(directory_path, 'neighbors.csv'))

    with open(os.path.join(directory_path, 'singles.pickle'), 'rb') as f:
        singles = pickle.load(f)

    with open(os.path.join(directory_path, 'repeats.pickle'), 'rb') as f:
        repeats = pickle.load(f)

    morphology_creator = MorphologyCreator()
    single_df, repeat_df, neigh_df = morphology_creator.run(singles, repeats, abs_df, neighbors_df, waves)

    single_df.to_csv(os.path.join(directory_path, 'singles.csv'), index=False)
    repeat_df.to_csv(os.path.join(directory_path, 'repeats.csv'), index=False)
    neighbors_df.to_csv(os.path.join(directory_path, 'neighbors.csv'), index=False)


def debug():
    args = parse_args()

    root_dir = r'C:\Users\Wojtek\Documents\Doktorat\Astral\data'
    directory = 'Cont_AA_1_2'
    directory_path = os.path.join(root_dir, directory)
    abs_df = pd.read_hdf(os.path.join(directory_path, 'segmentation_absolute.h5'))
    waves = np.load(os.path.join(directory_path, 'waves_morph.npy'))
    neighbors_df = pd.read_csv(os.path.join(directory_path, 'neighbors.csv'))

    with open(r'C:\Users\Wojtek\Documents\Doktorat\Astral\data\Cont_AA_1_2\singles.pickle', 'rb') as f:
        singles = pickle.load(f)

    with open(r'C:\Users\Wojtek\Documents\Doktorat\Astral\data\Cont_AA_1_2\repeats.pickle', 'rb') as f:
        repeats = pickle.load(f)

    morphology_creator = MorphologyCreator()
    single_df, repeat_df, neigh_df = morphology_creator.run(singles, repeats, abs_df, neighbors_df, waves)

    single_df.to_csv(os.path.join(directory_path, 'singles.csv'), index=False)
    repeat_df.to_csv(os.path.join(directory_path, 'repeats.csv'), index=False)
    neighbors_df.to_csv(os.path.join(directory_path, 'neighbors.csv'), index=False)


if __name__ == '__main__':
    main()
