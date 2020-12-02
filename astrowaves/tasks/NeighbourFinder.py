import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist
from skimage.morphology import binary_erosion
import cv2
from scipy.spatial import cKDTree
from collections import defaultdict
from scipy import ndimage
from operator import add
import argparse


class NeighbourFinder():

    def __init__(self):
        pass

    def _get_z_projection(self, shape_id, abs_df):
        proj = np.unique(abs_df.loc[abs_df['id'] == shape_id, ['x', 'y']].values.astype('int16'), axis=0)
        return proj

    def run(self, tolerance_xy, tolerance_z, absolute_df, dimensions_df):

        self.tolerance_xy = tolerance_xy
        self.tolerance_z = tolerance_z
        self.dimensions_df = dimensions_df
        self.absolute_df = absolute_df

        dist_dict = self.get_candidate_neighbors_dict(dimensions_df, absolute_df, tolerance_xy, tolerance_z)
        dist_dict = self.filter_distant_neighbors(dist_dict, tolerance_xy, tolerance_z)

        dist_df = pd.DataFrame(columns=['shape_id_1', 'shape_id_2', 'center_dist_xy',
                                        'center_dist_t', 'center_of_mass_dist_xy', 'center_of_mass_dist_t'])

        for shape_id, neighbor_dict in tqdm(dist_dict.items()):
            shape1 = shape_id
            for shape2 in neighbor_dict.keys():
                center_dist_xy, center_dist_z = self.calculate_euc_dists(dimensions_df, shape1, shape2)

                com_dist_xy, com_dist_t = self.calculate_com_dists(dimensions_df, absolute_df, shape1, shape2)

                row = {
                    'shape_id_1': shape1,
                    'shape_id_2': shape2,
                    'center_dist_xy': center_dist_xy,
                    'center_dist_t': center_dist_z,
                    'center_of_mass_dist_xy': com_dist_xy,
                    'center_of_mass_dist_t': com_dist_t
                }

                dist_df = dist_df.append(row, ignore_index=True)
        dist_df = dist_df.astype('int')
        return dist_df

    def get_tolerance_bounding_box(self, shape, tolerance_xy, tolerance_z):

        xl = shape['x_min'].values[0] - tolerance_xy
        yl = shape['y_min'].values[0] - tolerance_xy
        zl = shape['z_min'].values[0] - tolerance_z

        xu = shape['x_max'].values[0] + tolerance_xy
        yu = shape['y_max'].values[0] + tolerance_xy
        zu = shape['z_max'].values[0] + tolerance_z

        return (xl, xu), (yl, yu), (zl, zu)

    def get_neighbor_shapes(self, shape, tolerance_xy, tolerance_z):

        xb, yb, zb = self.get_tolerance_bounding_box(shape, tolerance_xy, tolerance_z)

        neighbor_shapes = self.dimensions_df.loc[(self.dimensions_df['x_min'].between(
            *xb)) | (self.dimensions_df['x_max'].between(*xb))]
        neighbor_shapes = neighbor_shapes.loc[(neighbor_shapes['y_min'].between(*yb))
                                              | (neighbor_shapes['y_max'].between(*yb))]
        neighbor_shapes = neighbor_shapes.loc[(neighbor_shapes['z_min'].between(*zb))
                                              | (neighbor_shapes['z_max'].between(*zb))]

        return neighbor_shapes

    def get_candidate_neighbors_dict(self, ddf, adf, tolerance_xy, tolerance_z):

        min_dist_dict = defaultdict(dict)

        ids = np.unique(ddf.id.values)

        for i in tqdm(ids):

            shape = ddf.loc[ddf['id'] == i]

            candidate_neighbors = self.get_neighbor_shapes(shape, tolerance_xy, tolerance_z)

            candidate_ids = np.unique(candidate_neighbors.id.values)

            for j in candidate_ids:
                if i != j:
                    shape1 = adf.loc[adf['id'] == i]
                    shape2 = adf.loc[adf['id'] == j]
                    shape1 = shape1[['x', 'y', 'z']]
                    shape2 = shape2[['x', 'y', 'z']]
                    # border1 = get_border_inds(shape1)
                    # border2 = get_border_inds(shape2)

                    shape1_xy = shape1[['x', 'y']]
                    shape2_xy = shape2[['x', 'y']]

                    shape1_z = np.unique(shape1[['z']].values)
                    shape2_z = np.unique(shape2[['z']].values)

                    shape1_z = np.expand_dims(shape1_z, 1)
                    shape2_z = np.expand_dims(shape2_z, 1)

                    min_dists_xy, min_dist_idx_xy = cKDTree(shape1_xy).query(shape2_xy, 1)
                    min_dists_z, min_dist_idx_z = cKDTree(shape1_z).query(shape2_z, 1)
        #            min_dists, min_dist_idx = cKDTree(border1).query(border2, 1)
                    #min_dists = (min_dists_xy.min(), min_dists_z.min())

                    min_dist_dict[i][j] = (min_dists_xy.min(), min_dists_z.min())
                    min_dist_dict[j][i] = (min_dists_xy.min(), min_dists_z.min())
        return min_dist_dict

    def filter_distant_neighbors(self, dist_dict, tolerance_xy, tolerance_z):
        for shape_id in dist_dict.keys():
            candidate_neighbor_dict = dist_dict[shape_id]
            filtered_candidate_dict = dict(
                filter(
                    lambda y: y[1][0] < tolerance_xy and y[1][1] < tolerance_z,
                    candidate_neighbor_dict.items()))
            dist_dict[shape_id] = filtered_candidate_dict
        return dist_dict

    def calculate_euc_dists(self, ddf, shape1, shape2):
        xy_axis1 = ddf.loc[ddf.id == shape1, ['center_y', 'center_x']].values
        xy_axis2 = ddf.loc[ddf.id == shape2, ['center_y', 'center_x']].values
        center_dist_xy = cdist(xy_axis1, xy_axis2)[0][0]

        z_axis1 = ddf.loc[ddf.id == shape1, ['center_z']].values
        z_axis1 = np.vstack([z_axis1, np.zeros((z_axis1.shape[0]))]).T
        z_axis2 = ddf.loc[ddf.id == shape2, ['center_z']].values
        z_axis2 = np.vstack([z_axis2, np.zeros((z_axis2.shape[0]))]).T

        center_dist_z = cdist(z_axis1, z_axis2)[0][0]

        return center_dist_xy, center_dist_z

    def calculate_com_dists(self, ddf, adf, shape1_id, shape2_id):

        shape1 = adf.loc[adf['id'] == shape1_id]
        shape2 = adf.loc[adf['id'] == shape2_id]
        shapes = [shape1, shape2]

        shapes = list(map(lambda df: df[['x', 'y', 'z']], shapes))

        offsets = []

        coms = []

        for shape in shapes:
            offsets.append([shape.x.min(), shape.y.min(), shape.z.min()])
            shape.x = shape.x - shape.x.min()
            shape.y = shape.y - shape.y.min()
            shape.z = shape.z - shape.z.min()

            indices = shape.values

            shape_np = np.zeros(((indices[:, 0].max() + 1, indices[:, 1].max() + 1, indices[:, 2].max() + 1)))

            for i in range(indices.shape[0]):
                shape_np[indices[i, 0], indices[i, 1], indices[i, 2]] = 1

            # print(np.unique(shape_np))
            com = ndimage.measurements.center_of_mass(shape_np)
            # print(com)
            com = list(map(lambda x: int(x), com))
            coms.append(com)

        coms_offset = []

        for com, offset in zip(coms, offsets):
            coms_offset.append(list(map(add, com, offset)))

        xy1, xy2 = coms_offset[0][:2], coms_offset[1][:2]
        xy1, xy2 = np.array(xy1), np.array(xy2)
        xy1, xy2 = np.expand_dims(xy1, -1).T, np.expand_dims(xy2, -1).T
        z1, z2 = coms_offset[0][2], coms_offset[1][2]

        com_dist_z = abs(z1-z2)
        com_dist_xy = cdist(xy1, xy2)[0][0]

        return com_dist_xy, com_dist_z

    def generate_df(self, dist_dict):
        dist_df = pd.DataFrame(columns=['shape_id_1', 'shape_id_2', 'center_dist_xy',
                                        'center_dist_t', 'center_of_mass_dist_xy', 'center_of_mass_dist_t'])

        for shape_id, neighbor_dict in tqdm(dist_dict.items()):
            shape1 = shape_id
            for shape2 in neighbor_dict.keys():
                center_dist_xy, center_dist_z = self.calculate_euc_dists(ddf, shape1, shape2)

                com_dist_xy, com_dist_z = self.calculate_com_dists(ddf, adf, shape1, shape2)

                row = {
                    'shape_id_1': shape1,
                    'shape_id_2': shape2,
                    'center_dist_xy': center_dist_xy,
                    'center_dist_t': center_dist_z,
                    'center_of_mass_dist_xy': com_dist_xy,
                    'center_of_mass_dist_t': com_dist_z
                }

                dist_df = dist_df.append(row, ignore_index=True)
        dist_df = dist_df.astype('int')
        return dist_df


def parse_args():
    parser = argparse.ArgumentParser(prog='Segmenter')
    parser.add_argument('--directory', help='output_directory')
    parser.add_argument('--rootdir', type=str, default='/app/data', help='root directory of files')
    parser.add_argument('--tolerance_xy', help='output_directory')
    parser.add_argument('--tolerance_t', help='output_directory')
    return parser.parse_args()


def main():
    args = parse_args()
    tolerance_xy = int(args.tolerance_xy)
    tolerance_z = int(args.tolerance_t)
    directory = args.directory
    root_dir = args.rootdir
    path = os.path.join(root_dir, directory)

    absolute_df = pd.read_hdf(os.path.join(path, 'segmentation_absolute.h5'))
    dims_df = pd.read_hdf(os.path.join(path, 'segmentation_dims.h5'))
    nfinder = NeighbourFinder()
    dict_df = nfinder.run(tolerance_xy, tolerance_z, absolute_df, dims_df)
    dict_df = dict_df.sort_values(by=['shape_id_1'])
    dict_df.to_csv(os.path.join(path, 'neighbors.csv'), index=False)

    ne2 = dict_df.groupby('shape_id_1').count().iloc[:, :1]
    ne3 = dict_df.groupby('shape_id_1').mean().iloc[:, -2:]

    neighbors_stat_df = pd.merge(ne2, ne3, right_index=True, left_index=True)
    neighbors_stat_df.index.Name = 'id'
    neighbors_stat_df.columns = ['n_neighbors', 'avg_xy_dist_center-of-mass', 'avg_t_interval_center-of-mass']
    neighbors_stat_df['id'] = neighbors_stat_df.index
    neighbors_stat_df = neighbors_stat_df[['id', 'n_neighbors',
                                           'avg_xy_dist_center-of-mass', 'avg_t_interval_center-of-mass']]
    neighbors_stat_df.to_csv(os.path.join(path, 'neighbors_statistics.csv'), index=False)


def debug():
    args = parse_args()

    directory = 'Cont_AN_2_4'
    root_dir = r'C:\Users\Wojtek\Documents\Doktorat\Astral\data'
    path = os.path.join(root_dir, directory)

    absolute_df = pd.read_hdf(os.path.join(path, 'segmentation_absolute.h5'))
    dims_df = pd.read_hdf(os.path.join(path, 'segmentation_dims.h5'))

    nfinder = NeighbourFinder()
    dict_df = nfinder.run(50, 100, absolute_df, dims_df)
    dict_df = dict_df.sort_values(by=['shape_id_1'])
    dict_df.to_csv(os.path.join(path, 'neighbors.csv'), index=False)

    ne2 = dict_df.groupby('shape_id_1').count().iloc[:, :1]
    ne3 = dict_df.groupby('shape_id_1').mean().iloc[:, -2:]

    neighbors_stat_df = pd.merge(ne2, ne3, right_index=True, left_index=True)
    neighbors_stat_df.index.Name = 'id'
    neighbors_stat_df.columns = ['n_neighbors', 'avg_xy_dist_center-of-mass', 'avg_t_interval_center-of-mass']
    neighbors_stat_df['id'] = neighbors_stat_df.index
    neighbors_stat_df = neighbors_stat_df[['id', 'n_neighbors',
                                           'avg_xy_dist_center-of-mass', 'avg_t_interval_center-of-mass']]
    neighbors_stat_df.to_csv(os.path.join(path, 'neighbors_statistics.csv'), index=False)


if __name__ == '__main__':
    main()
