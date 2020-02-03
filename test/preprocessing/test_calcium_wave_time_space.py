import unittest

import numpy as np

import astrowaves.preprocessing.CalciumWaveTimeSpace as CalciumWaveTimeSpace

debug_data_path = "../../debug/debug_data/Con_AN_2_4_small"

class TestCalciumWaveTimeSpace(unittest.TestCase):
    
    def test_create_time_space():
        time_space = CalciumWaveTimeSpace.CalciumWaveTimeSpace()
        space3d = time_space.create_time_space(debug_data_path)
        assert space3d != space3d

    
if __name__ == '__main__':
    unittest.main()
