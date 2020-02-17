import unittest

import numpy as np

from astrowaves.helpers.Config import Config

class ConfigTest(unittest.TestCase):
    
    def test_config_parsing(self):
        config = Config(path='cfg/cfg_test.xml')
        self.assertEqual(config.input_path, 'D:\\Doktorat\\Essen\\Con_AN_2_4\\')
        self.assertEqual(config.time_interval_ms, 300)
        self.assertEqual(config.std_threshold, 2.5)
        self.assertEqual(config.output_path, 'D:\\Doktorat\\Essen\\outputs\\1\\')

    
if __name__ == '__main__':
    unittest.main()
