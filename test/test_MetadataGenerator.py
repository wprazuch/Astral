import os

import pytest
from astrowaves.tasks.MetadataGenerator import generate_metadata

input_path1 = r'examples\\Cont_AA_2_1_small'
output_path1 = r'examples\\tmp'


@pytest.fixture
def input_path():
    return input_path1


@pytest.fixture
def output_path():
    return output_path1


# def test_metadata_generation(input_path, output_path):
#     generate_metadata(input_path, output_path)
#     assert os.path.exists(os.path.join(output_path, 'segmentation_absolute.h5'))
#     assert os.path.exists(os.path.join(output_path, 'segmentation_dims.h5'))
#     assert os.path.exists(os.path.join(output_path, 'segmentation_relative.h5'))
