import unittest

import numpy as np

from astrowaves.preprocessing.CalciumWaveTimeSpaceCreator import CalciumWaveTimeSpaceCreator


class TestCalciumWavesExtractor(unittest.TestCase):
    
    def test_create_time_space(self):
        
        timespace_creator = CalciumWaveTimeSpaceCreator()
        space3d = timespace_creator.run(input_path)
        assert space3d.shape == (608, 960, 31)

    
if __name__ == '__main__':
    unittest.main()
