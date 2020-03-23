import numpy as np
import pandas as pd
import LuigiConfig
import luigi
from astrowaves.preprocessing.TiffSplitter import TiffSplitter

class SplitTiff(luigi.Task):
        
    def run(self):
        self.luigi_config = LuigiConfig.LuigiConfig()
        splitter = TiffSplitter(input_path=self.luigi_config.input_path, output_path=self.luigi_config.output_path)
        splitter.run()

if __name__ == '__main__':
    luigi.build([SplitTiff()])