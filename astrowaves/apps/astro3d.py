from astrowaves.preprocessing.CalciumWaveTimeSpace import CalciumWaveTimeSpace
import xml.etree.ElementTree as ET
import os
import numpy as np

if __name__ == "__main__":

    cwts = CalciumWaveTimeSpace()

    timespace = cwts.create_time_space(path)

    np.save(output_path+"timespace1.npy", timespace)

    timespace = cwts.create_calcium_waves_map(timespace)
    timespace = timespace.astype('uint8')

    np.save(output_path+"waves_norm.npy", timespace)

