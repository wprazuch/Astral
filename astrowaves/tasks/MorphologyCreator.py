from scipy.ndimage import zoom
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
import math
from skimage.measure import regionprops
from .models.rows import RepeatsRow, SinglesRow
from .. import config


class MorphologyCreator:
    def get_shape_voxels_by_id(self, abs_csv, shape_id):
        """Gets the records of the given shape identifier

        Parameters
        ----------
        abs_csv : pd.DataFrame
            dataframe of absolute values in the timelapse
        shape_id : int
            id of the shape

        Returns
        -------
        pd.DataFrame
            dataframe with the records of a given shape only
        """
        return abs_csv.loc[abs_csv["id"] == shape_id]

    def get_shape_bbox(self, abs_csv, waves, shape_id):
        """Gets the bounding box of a given shape

        Parameters
        ----------
        abs_csv : pd.DataFrame
            dataframe of absolute values in the timelapse
        waves : np.ndarray
            timelapse of events
        shape_id : int
            id of the shape

        Returns
        -------
        np.ndarray
            bounding box of the segmentation only
        """
        abs_shape = abs_csv.loc[abs_csv["id"] == shape_id]
        min_x, min_y, min_z = (
            abs_shape["x"].min(),
            abs_shape["y"].min(),
            abs_shape["z"].min(),
        )
        max_x, max_y, max_z = (
            abs_shape["x"].max(),
            abs_shape["y"].max(),
            abs_shape["z"].max(),
        )
        segmentation = waves[min_y : max_y + 1, min_x : max_x + 1, min_z : max_z + 1]
        return segmentation.astype("uint8")

    def calculate_max_dims(self, segmentation):
        """Calculates maximum dimensions of the given segmentation

        Parameters
        ----------
        segmentation : np.ndarray
            bounding box with shape mask

        Returns
        -------
        Tuple[int]
            maximal dimensions of a given shape
        """
        max_y = np.sum(segmentation, axis=0).max()
        max_x = np.sum(segmentation, axis=1).max()
        max_z = np.sum(segmentation, axis=2).max()

        return max_y, max_x, max_z

    def calculate_morphology(self, shape, segmentation):
        max_y = np.sum(segmentation, axis=0).max()
        max_x = np.sum(segmentation, axis=1).max()
        max_z = np.sum(segmentation, axis=2).max()

        shape[shape == 1] = 255
        shape = zoom(shape, (0.5, 0.5, 0.5))

        sitk_img = sitk.GetImageFromArray(shape)
        sitk_mask = sitk.GetImageFromArray(segmentation)
        rs = RadiomicsShape(sitk_img, sitk_mask)
        sphericity = rs.getSphericityFeatureValue()
        sphericity = round(sphericity, 2)
        morph_dict = {
            "max_x_size": max_x,
            "max_y_size": max_y,
            "max_z_size": max_z,
            "sphericity": sphericity,
        }

        return morph_dict

    def create_3d_shape(self, shape_idxs):
        dim_x = shape_idxs[:, 0].max() + 1 - shape_idxs[:, 0].min()
        dim_y = shape_idxs[:, 1].max() + 1 - shape_idxs[:, 1].min()
        dim_t = shape_idxs[:, 2].max() + 1 - shape_idxs[:, 2].min()

        volume = np.zeros(shape=(dim_x, dim_y, dim_t))
        for row in shape_idxs:
            volume[row[0], row[1], row[2]] = 1

        return volume

    def calculate_morphology_for_singles(self, singles, abs_csv, waves, rel_df):
        """Gets morphological data for single events

        Parameters
        ----------
        singles : List[int]
            list of ids for single events
        abs_csv : pd.DataFrame
            dataframe of absolute values in a timelapse
        waves : np.ndarray
            3D timelapse of events
        rel_df : pd.DataFrame
            dataframe of relative values in a timelapse

        Returns
        -------
        pd.DataFrame
            dataframe of morhological statistics for each single event
        """

        morphology_data = []

        for single_id in singles:
            seg = self.get_shape_bbox(abs_csv, waves, single_id)
            max_y_size, max_x_size, max_z_size = self.calculate_max_dims(seg.copy())
            shape_id = int(single_id)

            shape2 = self.get_shape_voxels_by_id(rel_df, single_id)
            idxs = shape2[["x", "y", "z"]].values
            vol = self.create_3d_shape(idxs)
            proj = np.sum(vol, axis=2)
            proj[proj > 0] = 1
            proj = proj.astype(int)

            def circularity_fn(r):
                return (4 * math.pi * r.area) / (r.perimeter * r.perimeter)

            reg = list(regionprops(proj))
            max_xy_diameter = round(np.mean(reg[0].major_axis_length), 2)
            circularity = round(np.mean(circularity_fn(reg[0])), 2)

            single_row_obj = SinglesRow(
                shape_id,
                max_x_size,
                max_y_size,
                max_z_size,
                max_xy_diameter,
                circularity,
            )

            single_row = single_row_obj.get_row()

            morphology_data.append(single_row)

        morphology_df = pd.DataFrame(
            columns=config.SINGLES_DF_COLUMNS.values(), data=morphology_data
        )

        return morphology_df

    def calculate_morphology_for_repeats(self, repeats, abs_csv, waves, rel_df):
        """Calculates morphological statistics for repeated events

        Parameters
        ----------
        repeats : List[int]
            identifiers of repeated shapes
        abs_csv : pd.DataFrame
            dataframe of absolute values in the timelapse
        waves : np.ndarray
            timelapse of events
        rel_df : pd.DataFrame
            dataframe of relative values in the timelapse

        Returns
        -------
        pd.DataFrame
            dataframe of morphological statistics for the repeats
        """

        repeats_data = []

        for repeat_series in repeats:

            z_tuples = []

            max_x_sizes = []
            max_y_sizes = []
            max_z_sizes = []
            circularities = []
            max_xys = []

            for rep_id in repeat_series:
                shape = self.get_shape_voxels_by_id(abs_csv, rep_id)
                min_z, max_z = shape["z"].min(), shape["z"].max()
                z_tuples.append((min_z, max_z))
                seg = self.get_shape_bbox(abs_csv, waves, rep_id)
                max_y_size, max_x_size, max_z_size = self.calculate_max_dims(seg.copy())

                shape2 = self.get_shape_voxels_by_id(rel_df, rep_id)
                idxs = shape2[["x", "y", "z"]].values
                vol = self.create_3d_shape(idxs)
                proj = np.sum(vol, axis=2)
                proj[proj > 0] = 1
                proj = proj.astype(int)

                def circularity_fn(r):
                    return (4 * math.pi * r.area) / (r.perimeter * r.perimeter)

                reg = list(regionprops(proj))

                max_xy = reg[0].major_axis_length
                circularity = circularity_fn(reg[0])

                max_x_sizes.append(max_x_size)
                max_y_sizes.append(max_y_size)
                max_z_sizes.append(max_z_size)
                circularities.append(circularity)
                max_xys.append(max_xy)

            z_tuples = sorted(z_tuples, key=lambda x: x[0])
            dists = [
                abs(z_tuples[i + 1][0] - z_tuples[i][1])
                for i in range(0, len(z_tuples) - 1)
            ]

            centers = sorted([np.mean(z_extrema) for z_extrema in z_tuples])
            center_dists = [
                centers[i + 1] - centers[i] for i in range(len(centers) - 1)
            ]

            shape_ids = str(sorted(repeat_series))[1:-1].replace(",", "_")

            number_of_repeats = len(repeat_series)
            mean_max_x_size = round(np.mean(max_x_sizes), 2)
            mean_max_y_size = round(np.mean(max_y_sizes), 2)
            mean_max_z_size = round(np.mean(max_z_sizes), 2)
            med_inter_rep_min_z_dist = np.median(dists)
            med_inter_repeat_center_dist = np.median(center_dists)
            avg_circularity = round(np.mean(circularities), 2)
            avg_max_xy = round(np.mean(max_xys), 2)

            repeats_row_obj = RepeatsRow(
                shape_ids,
                number_of_repeats,
                mean_max_x_size,
                mean_max_y_size,
                mean_max_z_size,
                avg_max_xy,
                med_inter_rep_min_z_dist,
                med_inter_repeat_center_dist,
                avg_circularity,
            )

            repeats_row = repeats_row_obj.get_row()

            repeats_data.append(repeats_row)

        repeat_df = pd.DataFrame(
            columns=config.REPEATS_DF_COLUMNS.values(), data=repeats_data
        )
        return repeat_df

    def run(self, singles, repeats, abs_csv, neigh_csv, waves, rel_df):
        repeat_df = self.calculate_morphology_for_repeats(
            repeats, abs_csv, waves, rel_df
        )
        single_df = self.calculate_morphology_for_singles(
            singles, abs_csv, waves, rel_df
        )

        ids_to_delete = [item[1:] for item in repeats]
        ids_to_delete = [item for reps in ids_to_delete for item in reps]
        neigh_csv = neigh_csv.loc[(~neigh_csv["shape_id_1"].isin(ids_to_delete))]
        neigh_csv = neigh_csv.loc[(~neigh_csv["shape_id_2"].isin(ids_to_delete))]

        return single_df, repeat_df, neigh_csv


def parse_args():
    parser = argparse.ArgumentParser(prog="Segmenter")
    parser.add_argument("--directory", help="output_directory")
    parser.add_argument(
        "--rootdir", type=str, default="/app/data", help="root directory of files"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = args.rootdir
    directory = args.directory
    input_path = os.path.join(root_dir, directory)

    abs_df = pd.read_hdf(os.path.join(input_path, "segmentation_absolute.h5"))
    rel_df = pd.read_hdf(os.path.join(input_path, "segmentation_relative.h5"))
    waves = np.load(os.path.join(input_path, "labelled_waves.npy"))
    waves = waves.astype(bool)
    neighbors_df = pd.read_csv(os.path.join(input_path, "neighbors.csv"))

    with open(os.path.join(input_path, "singles.pickle"), "rb") as f:
        singles = pickle.load(f)

    with open(os.path.join(input_path, "repeats.pickle"), "rb") as f:
        repeats = pickle.load(f)

    morphology_creator = MorphologyCreator()
    single_df, repeat_df, neigh_df = morphology_creator.run(
        singles, repeats, abs_df, neighbors_df, waves, rel_df
    )

    single_df.to_csv(os.path.join(input_path, "singles.csv"), index=False)
    repeat_df.to_csv(os.path.join(input_path, "repeats.csv"), index=False)
    neighbors_df.to_csv(os.path.join(input_path, "neighbors.csv"), index=False)


def create_morphologies(input_path, output_path):
    abs_df = pd.read_hdf(os.path.join(input_path, "segmentation_absolute.h5"))
    rel_df = pd.read_hdf(os.path.join(input_path, "segmentation_relative.h5"))
    waves = np.load(os.path.join(input_path, "labelled_waves.npy"))
    waves = waves.astype(bool)
    neighbors_df = pd.read_csv(os.path.join(input_path, "neighbors.csv"))

    with open(os.path.join(input_path, "singles.pickle"), "rb") as f:
        singles = pickle.load(f)

    with open(os.path.join(input_path, "repeats.pickle"), "rb") as f:
        repeats = pickle.load(f)

    morphology_creator = MorphologyCreator()
    single_df, repeat_df, neigh_df = morphology_creator.run(
        singles, repeats, abs_df, neighbors_df, waves, rel_df
    )

    single_df.to_csv(os.path.join(output_path, "singles.csv"), index=False)
    repeat_df.to_csv(os.path.join(output_path, "repeats.csv"), index=False)
    neighbors_df.to_csv(os.path.join(output_path, "neighbors.csv"), index=False)


if __name__ == "__main__":
    main()
