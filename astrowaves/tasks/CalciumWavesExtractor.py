import pandas as pd
import numpy as np
import os
import logging
from tqdm import tqdm
import argparse
import math


class CalciumWavesExtractor:
    def remove_background(self, timespace):
        """Removes background offset from the timelapse to perform calcium wave extraction

        Parameters
        ----------
        timespace : np.ndarray
            3D timelapse of calcium events

        Returns
        -------
        np.ndarray
            3D timelapse of calcium events subtracted by the background intensity
        """
        background = np.mean(timespace, axis=2).astype("int16")
        background = background.reshape(background.shape + (1,))
        timespace = timespace - background
        timespace[timespace < 0] = 0
        return timespace

    def run(self, timespace):
        """Main function to perform wave extraction

        Parameters
        ----------
        timespace : np.ndarray
            3D timelapse of calcium events

        Returns
        -------
        np.ndarray
            3D timelapse of just extracted calcium events
        """
        logging.info(
            "Extracting calcium waves from 3d representation - might take a while..."
        )
        timespace = self.remove_background(timespace)
        minn = np.min(timespace).astype("uint16")
        ptp = np.ptp(timespace).astype("uint16")
        chunksize = 250
        iters = int(math.ceil(timespace.shape[2] / chunksize))
        for i in range(iters):
            # (255 * ((image_matrix - minn) / ptp)).astype('uint8')
            chunk = timespace[..., i * chunksize : (i + 1) * chunksize]
            chunk = (255 * ((chunk - minn) / ptp)).astype("uint8")
            timespace = timespace.astype("uint8")

        return timespace


def parse_args():
    parser = argparse.ArgumentParser(prog="timespacecreator")
    parser.add_argument("--directory", help="input path where images are stored")
    parser.add_argument(
        "--rootdir", type=str, default="/app/data", help="root directory of files"
    )
    args = parser.parse_args()
    return args


def generate_waves(input_timespace_npy_path, output_waves_npy_path):
    cwe = CalciumWavesExtractor()
    timespace = np.load(input_timespace_npy_path)
    waves = cwe.run(timespace)
    np.save(output_waves_npy_path, waves)


def main():
    args = parse_args()
    rootdir = args.rootdir
    directory = args.directory

    input_path = os.path.join(rootdir, directory)

    cwe = CalciumWavesExtractor()
    timespace = np.load(os.path.join(input_path, "timespace.npy"))
    waves = cwe.run(timespace)
    np.save(os.path.join(input_path, "waves.npy"), waves)


if __name__ == "__main__":
    main()
    # debug()
