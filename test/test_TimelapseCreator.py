import pytest
from astrowaves.tasks.TimelapseCreator import TimelapseCreator
import PIL
import numpy as np


def test_data_loading():

    timelapse_path = r'examples/Cont_AA_2_1.tif'
    timelapse_creator = TimelapseCreator()
    timelapse = timelapse_creator.load_timelapse(timelapse_path)

    assert isinstance(timelapse, PIL.TiffImagePlugin.TiffImageFile)


def test_3d_space_creation():
    timelapse_path = r'examples/Cont_AA_2_1.tif'
    timelapse_creator = TimelapseCreator()
    timelapse = timelapse_creator.load_timelapse(timelapse_path)
    timespace = timelapse_creator.create_3d_space(timelapse)

    assert isinstance(timespace, np.ndarray) and timespace.shape[2] == 1200
