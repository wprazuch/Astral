import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import os

class Config():

    def __init__(self, path : str = os.path.join('cfg', 'cfg.xml')):
        tree = ET.parse(path)
        root = tree.getroot()

        self.input_path = root.find('input_path').text
        self.time_interval_ms = int(root.find('time_interval_ms').text)
        self.std_threshold = float(root.find('standard_deviation_threshold').text)
        self.output_path = root.find('output_path').text





