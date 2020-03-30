import numpy as np
import pandas as pd
import os


class DataLoader():

    def __init__(self):
        pass

    def load_waves(self, path=None):
        waves = None
        if path is None:
            waves = np.load("debug\\waves.npy")
        else:
            waves = np.load(path)
        return waves
        