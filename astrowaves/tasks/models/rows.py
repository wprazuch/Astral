import os
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class DimensionsRow:

    id_: int = 0
    y_min: int = 0
    y_max: int = 0
    x_min: int = 0
    x_max: int = 0
    z_min: int = 0
    z_max: int = 0
    center_y: int = 0
    center_x: int = 0
    center_z: int = 0
    center_of_mass_cords: np.array = 0

    def get_row(self) -> List:
        row = [self.id_, self.y_min, self.y_max, self.x_min, self.x_max, self.z_min, self.z_max,
               self.center_y, self.center_x, self.center_z, *list(self.center_of_mass_cords)]

        return row
