import xml.etree.ElementTree as ET
import os
import numpy as np

from astrowaves.preprocessing.CalciumWaveTimeSpace import CalciumWaveTimeSpace
from astrowaves.helpers.Config import Config

if __name__ == "__main__":

    cwts = CalciumWaveTimeSpace()

    config = Config()

    timespace = cwts.create_time_space(config.input_path)

    np.save(config.output_path+"timespace.npy", timespace)

    timespace = cwts.create_calcium_waves_map(timespace)
    timespace = timespace.astype('uint8')

    np.save(config.output_path+"waves_norm.npy", timespace)

