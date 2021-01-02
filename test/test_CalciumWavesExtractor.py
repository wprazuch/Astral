import pytest
from astrowaves.tasks.CalciumWavesExtractor import CalciumWavesExtractor
import numpy as np


def test_background_removal():
    cwaves_extractor = CalciumWavesExtractor()

    random_matrix = np.random.randint(low=0, high=255, size=(200, 300, 600))
    background_removed = cwaves_extractor.remove_background(random_matrix)

    assert (background_removed >= 0).all()
