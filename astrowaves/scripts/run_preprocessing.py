import pandas as pd
import numpy as np
import os

from astrowaves.preprocessing import TiffSplitter, CalciumWaveTimeSpaceCreator, CalciumWavesExtractor

from astrowaves.helpers.Config import Config
import astrowaves.animations.animation_tools as anim_tools

if __name__ == "__main__":
    cfg = Config()

    debug_path = 'C:\\Users\\Wojtek\\Documents\\Doktorat\\AstrocyteCalciumWaveDetector\\debug'

    splitter = TiffSplitter.TiffSplitter()
    splitter.run(input_path=cfg.input_path, output_path=cfg.output_path)

    timespace_creator = CalciumWaveTimeSpaceCreator.CalciumWaveTimeSpaceCreator()
    timespace = timespace_creator.run(cfg.output_path)
    np.save(os.path.join(debug_path, "timespace.npy"), timespace)
    anim_tools.visualize_waves(timespace, filename="timespace.mp4")

    waves_extractor = CalciumWavesExtractor.CalciumWavesExtractor()
    waves = waves_extractor.run(timespace)
    anim_tools.visualize_waves(waves, filename="waves.mp4")
    np.save(os.path.join(debug_path, "waves.npy"), waves)


