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


@dataclass
class SinglesRow:
    shape_id: int
    max_x_size: float
    max_y_size: float
    max_z_size: float
    max_xy_diameter: float
    circularity: float

    def get_row(self) -> List:
        row = [self.shape_id, self.max_x_size, self.max_y_size,
               self.max_z_size, self.max_xy_diameter, self.circularity]

        return row


@dataclass
class RepeatsRow:
    shape_ids: list
    number_of_repeats: int
    avg_maximum_x: float
    avg_maximum_y: float
    avg_maximum_z: float
    avg_maximum_xy_diameter: float
    median_inter_repeat_min_z_dist: float
    median_inter_repeat_center_dist: float
    avg_circularity: float

    def get_row(self) -> List:
        row = [self.shape_ids, self.number_of_repeats, self.avg_maximum_x,
               self.avg_maximum_y, self.avg_maximum_z, self.avg_maximum_xy_diameter,
               self.median_inter_repeat_min_z_dist, self.median_inter_repeat_center_dist, self.avg_circularity]

        return row
