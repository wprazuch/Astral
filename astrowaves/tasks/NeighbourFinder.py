from itertools import product
from .. import config
from scipy.ndimage import binary_fill_holes
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
from joblib import Parallel, delayed


class NeighbourFinder:
    def _get_z_projection(self, shape_id, abs_df):
        """Gets the projection of a given shape on the time dimension

        Parameters
        ----------
        shape_id : int
            Identifier of the shape
        abs_df : pd.DataFrame
            dataframe with absolute values of the events

        Returns
        -------
        np.ndarray
            Coordinates of the xy axis in which the event occurs
        """
        proj = np.unique(
            abs_df.loc[abs_df["id"] == shape_id, ["x", "y"]].values.astype("int16"),
            axis=0,
        )
        return proj

    def get_bounding_box_neighbours(self, dimensions_df, tolerance_xy, tolerance_z):
        """Searches for neighbours in a given neighbourhood range

        Parameters
        ----------
        dimensions_df : pd.DataFrame
            dataframe of extreme coordinates of events
        tolerance_xy : float
            tolerance in a xy plane
        tolerance_z : float
            tolerance in time dimension

        Returns
        -------
        Dict
            Dict of neighbours for each shape
        """
        neighbours_dict = defaultdict(dict)
        ids = np.unique(dimensions_df.id.values)
        for i in tqdm(ids):

            shape = dimensions_df.loc[dimensions_df["id"] == i]
            candidate_neighbors = self.get_neighbor_shapes(
                shape, tolerance_xy, tolerance_z
            )
            candidate_ids = list(np.unique(candidate_neighbors.id.values))
            neighbours_dict[i] = candidate_ids[1:]

        return neighbours_dict

    def generate_neighbor_row(self, shape1_id, shape2_id, dims_df_chunk):
        """Calculates all the metrics necessary for one nieghbour row

        Parameters
        ----------
        shape1_id : int
            reference shape id
        shape2_id : int
            neighbour shape id
        dims_df_chunk : pd.DataFrame
            dataframe of dimensions

        Returns
        -------
        List
            list of values for the row
        """
        center_dist_xy, center_dist_z = self.calculate_euc_dists(
            dims_df_chunk, shape1_id, shape2_id
        )

        com_dist_xy, com_dist_t = self.calculate_center_of_mass_dists(
            dims_df_chunk, shape1_id, shape2_id
        )

        # row_dict = {
        #     'shape_id_1': shape1_id,
        #     'shape_id_2': shape2_id,
        #     'center_dist_xy': center_dist_xy,
        #     'center_dist_t': center_dist_z,
        #     'center_of_mass_dist_xy': com_dist_xy,
        #     'center_of_mass_dist_t': com_dist_t
        # }

        row = [
            shape1_id,
            shape2_id,
            center_dist_xy,
            center_dist_z,
            com_dist_xy,
            com_dist_t,
        ]

        return row

    def generate_neighbour_data_for(self, shape1_id, neighbour_ids):
        """Generates neighbor data for a given shape

        Parameters
        ----------
        shape1_id : int
            shape identifier
        neighbour_ids : List[int]
            list of neighbor identifiers

        Returns
        -------
        List
            list of neighbor rows for a given shape
        """

        dims_df_chunk = self.dimensions_df.loc[
            (self.dimensions_df["id"] == shape1_id)
            | (self.dimensions_df["id"].isin(neighbour_ids))
        ]

        shape1_id_row_data = []

        for shape2_id in neighbour_ids:
            shape1_id_row = self.generate_neighbor_row(
                shape1_id, shape2_id, dims_df_chunk
            )
            shape1_id_row_data.append(shape1_id_row)
        return shape1_id_row_data

    def run(self, tolerance_xy, tolerance_z, absolute_df, dimensions_df):

        self.tolerance_xy = tolerance_xy
        self.tolerance_z = tolerance_z
        self.dimensions_df = dimensions_df
        self.absolute_df = absolute_df

        neighbors_dict = self.get_bounding_box_neighbours(
            dimensions_df, tolerance_xy, tolerance_z
        )

        row_data = []

        for shape1, neighbour_ids in tqdm(neighbors_dict.items()):

            shape1_neighbor_data = self.generate_neighbour_data_for(
                shape1, neighbour_ids
            )

            row_data.extend(shape1_neighbor_data)

            # dist_df = dist_df.append(row_dict, ignore_index=True)

        dist_df = pd.DataFrame(
            columns=[
                "shape_id_1",
                "shape_id_2",
                "center_dist_xy",
                "center_dist_t",
                "center_of_mass_dist_xy",
                "center_of_mass_dist_t",
            ],
            data=row_data,
        )
        dist_df = dist_df.astype("int")
        dist_df = dist_df.sort_values(by=["shape_id_1"])

        return dist_df

    def get_tolerance_bounding_box(self, shape, tolerance_xy, tolerance_z):

        xl = shape["x_min"].values[0] - tolerance_xy
        yl = shape["y_min"].values[0] - tolerance_xy
        zl = shape["z_min"].values[0] - tolerance_z

        xu = shape["x_max"].values[0] + tolerance_xy
        yu = shape["y_max"].values[0] + tolerance_xy
        zu = shape["z_max"].values[0] + tolerance_z

        return (xl, xu), (yl, yu), (zl, zu)

    def get_neighbor_shapes(self, shape, tolerance_xy, tolerance_z):

        xb, yb, zb = self.get_tolerance_bounding_box(shape, tolerance_xy, tolerance_z)

        neighbor_shapes = self.dimensions_df.loc[
            (self.dimensions_df["x_min"].between(*xb))
            | (self.dimensions_df["x_max"].between(*xb))
        ]
        neighbor_shapes = neighbor_shapes.loc[
            (neighbor_shapes["y_min"].between(*yb))
            | (neighbor_shapes["y_max"].between(*yb))
        ]
        neighbor_shapes = neighbor_shapes.loc[
            (neighbor_shapes["z_min"].between(*zb))
            | (neighbor_shapes["z_max"].between(*zb))
        ]

        return neighbor_shapes

    def find_contours(self, shape_xy):
        max_y = shape_xy["y"].max() + 1
        max_x = shape_xy["x"].max() + 1

        placeholder = np.zeros((max_x, max_y))

        cords = shape_xy.values

        placeholder[cords[:, 0], cords[:, 1]] = 1

        placeholder = placeholder.astype(np.uint8)
        filled = binary_fill_holes(placeholder)
        filled = filled.astype(np.uint8)

        contours = cv2.findContours(filled, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours = contours[0]
        cnt = np.zeros(placeholder.shape)
        cnt[contours[:, 0, 1], contours[:, 0, 0]] = 1

        contour_cords = np.argwhere(cnt != 0)
        return contour_cords

    def get_candidate_neighbors_dict(self, ddf, adf, tolerance_xy, tolerance_z):

        min_dist_dict = defaultdict(dict)

        ids = np.unique(ddf.id.values)

        for i in tqdm(ids):

            shape = ddf.loc[ddf["id"] == i]

            candidate_neighbors = self.get_neighbor_shapes(
                shape, tolerance_xy, tolerance_z
            )

            candidate_ids = np.unique(candidate_neighbors.id.values)

            shape1 = adf.loc[adf["id"] == i]
            shape1 = shape1[["x", "y", "z"]]
            shape1_xy = shape1[["x", "y"]]
            shape1_xy = self.find_contours(shape1_xy)
            shape1_z = np.unique(shape1[["z"]].values)
            shape1_z = np.expand_dims(shape1_z, 1)

            for j in candidate_ids:
                if i != j:

                    shape2 = adf.loc[adf["id"] == j]

                    shape2 = shape2[["x", "y", "z"]]
                    # border1 = get_border_inds(shape1)
                    # border2 = get_border_inds(shape2)

                    shape2_xy = shape2[["x", "y"]]
                    shape2_xy = self.find_contours(shape2_xy)
                    shape2_z = np.unique(shape2[["z"]].values)
                    shape2_z = np.expand_dims(shape2_z, 1)

                    min_dists_xy, min_dist_idx_xy = cKDTree(shape1_xy).query(
                        shape2_xy, 1
                    )
                    min_dists_z, min_dist_idx_z = cKDTree(shape1_z).query(shape2_z, 1)
                    #            min_dists, min_dist_idx = cKDTree(border1).query(border2, 1)
                    # min_dists = (min_dists_xy.min(), min_dists_z.min())

                    min_dist_dict[i][j] = (min_dists_xy.min(), min_dists_z.min())
                    min_dist_dict[j][i] = (min_dists_xy.min(), min_dists_z.min())
        return min_dist_dict

    def filter_distant_neighbors(self, dist_dict, tolerance_xy, tolerance_z):
        for shape_id in dist_dict.keys():
            candidate_neighbor_dict = dist_dict[shape_id]
            filtered_candidate_dict = dict(
                filter(
                    lambda y: y[1][0] < tolerance_xy and y[1][1] < tolerance_z,
                    candidate_neighbor_dict.items(),
                )
            )
            dist_dict[shape_id] = filtered_candidate_dict
        return dist_dict

    def get_bounding_box_cords(self, dim_df_row):
        x_min = dim_df_row[config.DIMENSIONS_DF_COLUMNS["X_MIN"]].values[0]
        y_min = dim_df_row[config.DIMENSIONS_DF_COLUMNS["Y_MIN"]].values[0]
        x_max = dim_df_row[config.DIMENSIONS_DF_COLUMNS["X_MAX"]].values[0]
        y_max = dim_df_row[config.DIMENSIONS_DF_COLUMNS["Y_MAX"]].values[0]

        z_min = dim_df_row[config.DIMENSIONS_DF_COLUMNS["Z_MIN"]].values[0]
        z_max = dim_df_row[config.DIMENSIONS_DF_COLUMNS["Z_MAX"]].values[0]

        xy_cords = list(product([x_min, x_max], [y_min, y_max]))

        z_cords = [z_min, z_max]

        return xy_cords, z_cords

    def calculate_euc_dists(self, ddf, shape1, shape2):
        xy_axis1 = ddf.loc[ddf.id == shape1, ["center_y", "center_x"]].values
        xy_axis2 = ddf.loc[ddf.id == shape2, ["center_y", "center_x"]].values
        center_dist_xy = cdist(xy_axis1, xy_axis2)[0][0]

        z_axis1 = ddf.loc[ddf.id == shape1, ["center_z"]].values
        z_axis1 = np.vstack([z_axis1, np.zeros((z_axis1.shape[0]))]).T
        z_axis2 = ddf.loc[ddf.id == shape2, ["center_z"]].values
        z_axis2 = np.vstack([z_axis2, np.zeros((z_axis2.shape[0]))]).T

        center_dist_z = cdist(z_axis1, z_axis2)[0][0]

        return center_dist_xy, center_dist_z

    def calculate_center_of_mass_dists(self, ddf, shape1_id, shape2_id):

        z1 = ddf.loc[
            ddf[config.DIMENSIONS_DF_COLUMNS["ID"]] == shape1_id,
            config.DIMENSIONS_DF_COLUMNS["CENTER_OF_MASS_Z"],
        ].values
        z2 = ddf.loc[
            ddf[config.DIMENSIONS_DF_COLUMNS["ID"]] == shape2_id,
            config.DIMENSIONS_DF_COLUMNS["CENTER_OF_MASS_Z"],
        ].values

        xy1 = ddf.loc[
            ddf[config.DIMENSIONS_DF_COLUMNS["ID"]] == shape1_id,
            [
                config.DIMENSIONS_DF_COLUMNS["CENTER_OF_MASS_X"],
                config.DIMENSIONS_DF_COLUMNS["CENTER_OF_MASS_Y"],
            ],
        ].values

        xy2 = ddf.loc[
            ddf[config.DIMENSIONS_DF_COLUMNS["ID"]] == shape2_id,
            [
                config.DIMENSIONS_DF_COLUMNS["CENTER_OF_MASS_X"],
                config.DIMENSIONS_DF_COLUMNS["CENTER_OF_MASS_Y"],
            ],
        ].values

        com_dist_z = abs(z1 - z2)
        com_dist_xy = cdist(xy1, xy2)[0][0]

        return com_dist_xy, com_dist_z


def calculate_neighbors_statistics(dict_df):
    ne2 = dict_df.groupby("shape_id_1").count().iloc[:, :1]
    ne3 = dict_df.groupby("shape_id_1").mean().iloc[:, -2:]

    neighbors_stat_df = pd.merge(ne2, ne3, right_index=True, left_index=True)
    neighbors_stat_df.index.Name = "id"
    neighbors_stat_df.columns = [
        "n_neighbors",
        "avg_xy_dist_center-of-mass",
        "avg_t_interval_center-of-mass",
    ]
    neighbors_stat_df["id"] = neighbors_stat_df.index
    neighbors_stat_df = neighbors_stat_df[
        [
            "id",
            "n_neighbors",
            "avg_xy_dist_center-of-mass",
            "avg_t_interval_center-of-mass",
        ]
    ]

    return neighbors_stat_df


def find_neighbors(input_path, output_path, tolerance_xy=30, tolerance_z=20):
    absolute_df = pd.read_hdf(os.path.join(input_path, "segmentation_absolute.h5"))
    dims_df = pd.read_hdf(os.path.join(input_path, "segmentation_dims.h5"))
    nfinder = NeighbourFinder()
    dict_df = nfinder.run(tolerance_xy, tolerance_z, absolute_df, dims_df)
    dict_df.to_csv(os.path.join(output_path, "neighbors.csv"), index=False)

    neighbors_stat_df = calculate_neighbors_statistics(dict_df)
    neighbors_stat_df.to_csv(
        os.path.join(output_path, "neighbors_statistics.csv"), index=False
    )


def parse_args():
    parser = argparse.ArgumentParser(prog="Segmenter")
    parser.add_argument("--directory", help="output_directory")
    parser.add_argument(
        "--rootdir", type=str, default="/app/data", help="root directory of files"
    )
    parser.add_argument("--tolerance_xy", help="output_directory")
    parser.add_argument("--tolerance_t", help="output_directory")
    return parser.parse_args()


def main():
    args = parse_args()
    tolerance_xy = int(args.tolerance_xy)
    tolerance_z = int(args.tolerance_t)
    directory = args.directory
    root_dir = args.rootdir
    input_path = os.path.join(root_dir, directory)

    absolute_df = pd.read_hdf(os.path.join(input_path, "segmentation_absolute.h5"))
    dims_df = pd.read_hdf(os.path.join(input_path, "segmentation_dims.h5"))
    nfinder = NeighbourFinder()
    dict_df = nfinder.run(tolerance_xy, tolerance_z, absolute_df, dims_df)
    dict_df.to_csv(os.path.join(input_path, "neighbors.csv"), index=False)

    neighbors_stat_df = calculate_neighbors_statistics(dict_df)
    neighbors_stat_df.to_csv(
        os.path.join(input_path, "neighbors_statistics.csv"), index=False
    )


if __name__ == "__main__":
    main()
