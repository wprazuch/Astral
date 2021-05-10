import fire
import numpy as np
from czifile import imread as czi_read
from tiffile import imsave as tiffsave
from astrowaves.utils import to_255
from astrowaves.utils import generate_video
from pathlib import Path
from tiffile import imread as tiff_read
from tiffile import imsave as tiff_save
from astrowaves.utils import generate_video
import pandas as pd
from tqdm import tqdm

from scipy.fft import fft, ifft


def correct_intensities(array_3d, method="f0"):
    """Function for intensity correction

    Parameters
    ----------
    array_3d : np.ndarray
        timelapse to be intensity-corrected
    method : str, optional
        type of correction method, by default "f0"

    Returns
    -------
    np.ndarray
        Intensity corrected timelapse

    """

    if method == "f0":
        corrected_3d = f0_correction(array_3d)
    elif method == "pafft":
        corrected_3d = f0_correction(array_3d)
    else:
        raise AttributeError(
            f"Method {method} not implemented for intensity correction!"
        )

    return corrected_3d


def f0_correction(array_3d):
    """Performs f0 correction on the timelapse

    Parameters
    ----------
    array_3d : np.ndarray
        timelapse to be corrected

    Returns
    -------
    np.ndarray
        intensity corrected timelapse
    """
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


def find_minimum(current_segment, reference_segment):

    current_idxs = np.argsort(current_segment)
    reference_idxs = np.argsort(reference_segment)

    for i in range(len(current_segment // 20)):
        for j in range(len(reference_segment // 20)):
            if current_idxs[i] == reference_idxs[j]:
                return current_idxs[i]

    return current_idxs[0]


def fft_correlation(spectrum, target, shift):
    M = len(target)
    diff = 1000000000
    for i in range(20 - 1):
        cur_diff = 2 ** i - M
        if cur_diff > 0 and cur_diff < diff:
            diff = cur_diff

    # changed by adding + 1 - CAUTION
    padding = np.zeros(diff)
    target = np.hstack([target, padding])
    spectrum = np.hstack([spectrum, padding])

    M = M + diff
    X = fft(target)
    Y = fft(spectrum)

    R = X * np.conj(Y)
    R = R / M
    rev = ifft(R)

    vals = np.real(rev)
    max_position = 0
    maxi = -1

    if M < shift:
        shift = M

    for i in range(shift):
        if vals[i] > maxi:
            maxi = vals[i]
            max_position = i

        if vals[len(vals) - i] > maxi:
            maxi = vals[len(vals) - i]
            max_position = len(vals) - i

    if maxi < 0.1:
        lag = 0
        return lag

    # CHANGED BY DELETING - 1 - CAUTION
    if max_position > len(vals) / 2:
        lag = max_position - len(vals)
    else:
        lag = max_position

    return lag


def move(seg, lag):
    if lag == 0 or lag >= len(seg):
        return seg

    if lag > 0:
        ins = [1 * seg[0] for item in range(lag)]
        moved_seg = [*ins, *seg[: len(seg) - lag]]
    elif lag < 0:
        lag = np.abs(lag)
        ins = [1 * seg[-1] for item in range(lag)]
        moved_seg = [*seg[lag:], *ins]

    return moved_seg


def pafft_motion_correction(img):

    df = pd.DataFrame()

    for i in tqdm(range(img.shape[0])):
        uniq, cnts = np.unique(img[i], return_counts=True)
        row_dict = dict(zip(uniq, cnts))
        df = df.append(row_dict, ignore_index=True)

    pixel_values = sorted(list(df.columns.values.copy()))
    df_sorted = df[pixel_values]

    # fill nans with interpolations
    for i in range(df_sorted.shape[0]):

        df_sorted.iloc[i, :] = (
            df_sorted.iloc[i, :]
            .interpolate(method="nearest", axis=0)
            .ffill()
            .bfill()
            .values
        )

    mz_length = df.shape[0]

    shift_perc = 0.1
    scale = (shift_perc * 0.01 * mz_length) / (max(pixel_values) - min(pixel_values))

    seg_size = 200

    spectra = df_sorted.values.copy()

    no_frames = spectra.shape[0]
    no_pixel_values = spectra.shape[1]

    reference = spectra[0]

    aligned_spectrum = []
    lag_vectors = []

    for i in tqdm(range(1, no_frames, 1)):
        current_hist = spectra[i]

        start_position = 0
        aligned = []

        while_loop_execution_count = 0

        while start_position <= no_pixel_values:

            end_position = start_position + (seg_size * 2)

            if end_position >= no_pixel_values:
                samseg = spectra[i, start_position:].copy()
                refseg = reference[start_position:].copy()
            else:
                # deleting -1 does not change length of vector - CAUTION
                samseg = spectra[i, start_position + seg_size : end_position - 1].copy()
                refseg = reference[start_position + seg_size : end_position - 1].copy()
                min_position = find_minimum(samseg, refseg)
                end_position = start_position + min_position + seg_size
                samseg = spectra[i, start_position:end_position]
                refseg = reference[start_position:end_position]

            shift = int(scale * pixel_values[i + int(len(samseg) / 2)])
            lag = fft_correlation(samseg, refseg, shift)
            samseg_moved = move(samseg, lag)

            aligned.extend(samseg_moved)

            start_position = end_position

        aligned_spectrum.append(aligned)
        lag_vectors.append(lag)

    aligned_spectrum = np.array(aligned_spectrum)

    return aligned_spectrum


if __name__ == "__main__":
    fire.Fire()
