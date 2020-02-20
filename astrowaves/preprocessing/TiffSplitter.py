import numpy as np
from PIL import Image
import os

class TiffSplitter():

    def __init__(self):
        pass

    def run(self, input_path, output_path):
        img = Image.open(input_path)
        for i in range (1200):
            try:
                img.seek(i)
                img.save(os.path.join(output_path, 'image_%s.tif'%(i,)))
            except EOFError:
                return None
