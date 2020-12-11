import pytest
from astrowaves.tasks.TimelapseCreator import TimelapseCreator, generate_timespace
import PIL
import numpy as np
import os
from . import TEST_ARTIFACTS

file_paths = [f'examples/{test_artifact}/{test_artifact}.tif' for test_artifact in TEST_ARTIFACTS]
output_paths = [f'examples/{test_artifact}/timespace.npy' for test_artifact in TEST_ARTIFACTS]


@pytest.mark.parametrize('file_path', file_paths)
def test_data_loading(file_path):

    timelapse_path = file_path
    timelapse_creator = TimelapseCreator()
    timelapse = timelapse_creator.load_timelapse(timelapse_path)

    assert isinstance(timelapse, PIL.TiffImagePlugin.TiffImageFile)


@pytest.mark.parametrize('file_path', file_paths)
def test_3d_space_creation(file_path):
    timelapse_path = file_path
    timelapse_creator = TimelapseCreator()
    timelapse = timelapse_creator.load_timelapse(timelapse_path)
    timespace = timelapse_creator.create_3d_space(timelapse)

    assert isinstance(timespace, np.ndarray) and len(timespace.shape) == 3
