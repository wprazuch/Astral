import numpy as np
from PIL import Image
import os
from pathlib import Path

class TiffSplitter():

    def __init__(self, input_path=None, output_path=None):
        self.input_path = input_path
        self.output_path = output_path


    def run(self, input_path=None, output_path=None):
        if input_path is not None:
            self.input_path = input_path
        if output_path is not None:
            self.output_path = output_path

        self.__check_output_path()

        img = Image.open(input_path)
        for i in range (1200):
            try:
                img.seek(i)
                img.save(os.path.join(output_path, 'image_%s.tif'%(i,)))
            except EOFError:
                return None

    def __check_output_path(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)