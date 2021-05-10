import fire
import numpy as np
from czifile import imread as czi_read
from tiffile import imsave as tiffsave
from astrowaves.utils import to_255
from astrowaves.utils import generate_video
from pathlib import Path


def correct_intensities(array_3d, method="f0"):

    if method == "f0":
        corrected_3d = f0_correction(array_3d)
    else:
        raise AttributeError(
            f"Method {method} not implemented for intensity correction!"
        )

    return corrected_3d


def f0_correction(array_3d):
    means_ = []

    # Calculate average for each frame
    for i in range(array_3d.shape[0]):
        mean_ = np.mean(array_3d[i])
        means_.append(mean_)

    # Align means to the first frame of the sequence
    for i in range(1, array_3d.shape[0], 1):
        coefficient = means_[i] / means_[0]
        array_3d[i] = array_3d[i] / coefficient

    return array_3d


def perform_intensity_correction(
    input_czi_path, method="f0", output_file_path=None, debug=True
):

    if output_file_path is None:
        filepath = Path(input_czi_path)
        parent = filepath.parent
        suffix = filepath.suffix
        output_file_path = input_czi_path + ".tif"

    array_3d = czi_read(input_czi_path)
    array_3d = np.squeeze(array_3d)

    array_3d_corrected = correct_intensities(array_3d, method=method)
    array_3d_corrected = to_255(array_3d_corrected)
    array_3d_corrected = array_3d_corrected.astype("uint8")

    tiffsave(output_file_path, array_3d_corrected, photometric="minisblack")

    if debug == True:
        generate_video(array_3d, parent, "raw.mp4")
        generate_video(array_3d_corrected, parent, "intensity_corrected.mp4")


if __name__ == "__main__":
    fire.Fire()
