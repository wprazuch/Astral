import pytest
from astrowaves.tasks.TimelapseCreator import generate_timespace
from astrowaves.tasks.CalciumWavesExtractor import generate_waves
from astrowaves.tasks.MaskGenerator import generate_masks
from astrowaves.tasks.WaveLabeller import label_waves
from astrowaves.tasks.MetadataGenerator import generate_metadata
from astrowaves.tasks.CleanupMetadata import cleanup
from astrowaves.tasks.NeighbourFinder import find_neighbors
import PIL
import numpy as np
import os
from . import TEST_ARTIFACTS

file_paths = [f'examples/{test_artifact}/{test_artifact}.tif' for test_artifact in TEST_ARTIFACTS]
output_timespace_paths = [f'examples/{test_artifact}/timespace.npy' for test_artifact in TEST_ARTIFACTS]
output_waves_paths = [f'examples/{test_artifact}/waves.npy' for test_artifact in TEST_ARTIFACTS]
output_waves_binary_paths = [f'examples/{test_artifact}/labelled_waves.npy' for test_artifact in TEST_ARTIFACTS]
output_waves_inds_paths = [f'examples/{test_artifact}/waves_inds.pck' for test_artifact in TEST_ARTIFACTS]

example_dirs = [f'examples/{test_artifact}' for test_artifact in TEST_ARTIFACTS]


@pytest.mark.run(order=1)
@pytest.mark.parametrize('input_path,output_path',
                         [(input_path, output_path) for input_path, output_path in zip(
                             file_paths, output_timespace_paths)])
def test_timelapse_creator(input_path, output_path):
    generate_timespace(input_path, output_path)
    assert os.path.exists(output_path)


@pytest.mark.run(order=2)
@pytest.mark.parametrize('input_path,output_path',
                         [(input_path, output_path) for input_path, output_path in zip(
                             output_timespace_paths, output_waves_paths)])
def test_cacium_waves_extractor(input_path, output_path):
    generate_waves(input_path, output_path)
    assert os.path.exists(output_path)


@pytest.mark.run(order=3)
@pytest.mark.parametrize('input_path,output_path',
                         [(input_path, output_path) for input_path, output_path in zip(
                             output_waves_paths, output_waves_binary_paths)])
def test_mask_generator(input_path, output_path):
    generate_masks(input_path, output_path, x_range=3, y_range=3)
    assert os.path.exists(output_path)


@pytest.mark.run(order=4)
@pytest.mark.parametrize('input_path,output_path',
                         [(input_path, output_path) for input_path, output_path in zip(
                             output_waves_binary_paths, output_waves_inds_paths)])
def test_wave_labelling(input_path, output_path):
    label_waves(input_path, output_path, volume_threshold=25)
    assert os.path.exists(output_path)


@pytest.mark.run(order=5)
@pytest.mark.parametrize('input_path,output_path',
                         [(input_path, output_path) for input_path, output_path in zip(
                             example_dirs, example_dirs)])
def test_metadata_generation(input_path, output_path):
    generate_metadata(input_path, output_path)
    assert os.path.exists(os.path.join(output_path, 'segmentation_relative.h5'))
    assert os.path.exists(os.path.join(output_path, 'segmentation_absolute.h5'))
    assert os.path.exists(os.path.join(output_path, 'segmentation_dims.h5'))


@pytest.mark.run(order=6)
@pytest.mark.parametrize('input_path',
                         [input_path for input_path in example_dirs])
def test_cleanup(input_path):
    cleanup(input_path)
    assert not os.path.exists(os.path.join(input_path, 'waves_inds.pck'))


@pytest.mark.run(order=5)
@pytest.mark.parametrize('input_path,output_path',
                         [(input_path, output_path) for input_path, output_path in zip(
                             example_dirs, example_dirs)])
def test_neighbors_finding(input_path, output_path):
    find_neighbors(input_path, output_path)
    assert os.path.exists(os.path.join(output_path, 'segmentation_relative.h5'))
