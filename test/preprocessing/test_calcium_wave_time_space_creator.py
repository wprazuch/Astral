import unittest

import numpy as np

from astrowaves.preprocessing.CalciumWaveTimeSpaceCreator import CalciumWaveTimeSpaceCreator

debug_data_path = "../../debug/debug_data/Con_AN_2_4_small"

class TestCalciumWaveTimeSpace(unittest.TestCase):
    
    def test_create_time_space(self):
        input_path = "C:\\Users\\Wojtek\\Documents\\Doktorat\\AstrocyteCalciumWaveDetector\\test_files\\Con_AN_2_4"
        timespace_creator = CalciumWaveTimeSpaceCreator()
        space3d = timespace_creator.run(input_path)
        assert space3d.shape == (608, 960, 31)

    
if __name__ == '__main__':
    unittest.main()
