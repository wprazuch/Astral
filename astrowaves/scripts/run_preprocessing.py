import pandas as pd
import numpy as np
import os

from astrowaves.preprocessing import TiffSplitter, CalciumWaveTimeSpaceCreator, CalciumWavesExtractor

from astrowaves.helpers.Config import Config

if __name__ == "__main__":
    cfg = Config()

    splitter = TiffSplitter.TiffSplitter()
    splitter.run(input_path=cfg.input_path, output_path=cfg.output_path)

    timespace_creator = CalciumWaveTimeSpaceCreator.CalciumWaveTimeSpaceCreator()
    timespace_creator.run(path_to_image_sequence=cfg.output_path, output_path=cfg.temp_path)

    waves_extractor = CalciumWavesExtractor.CalciumWavesExtractor(output_path=cfg.temp_path)
    waves_extractor.run(input_file=str(os.path.join(cfg.temp_path, "timespace.npy")))
