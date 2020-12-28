import os


ABS_DF_COLUMNS = {
    'ID': 'id',
    'Y': 'y',
    'X': 'x',
    'Z': 'z',
    'COLOR': 'color'
}

REL_DF_COLUMNS = ABS_DF_COLUMNS.copy()


DIMENSIONS_DF_COLUMNS = {
    'ID': 'id',
    'Y_MIN': 'y_min',
    'Y_MAX': 'y_max',
    'X_MIN': 'x_min',
    'X_MAX': 'x_max',
    'Z_MIN': 'z_min',
    'Z_MAX': 'z_max',
    'CENTER_Y': 'center_y',
    'CENTER_X': 'center_x',
    'CENTER_Z': 'center_z',
    'CENTER_OF_MASS_X': 'center_of_mass_x',
    'CENTER_OF_MASS_Y': 'center_of_mass_y',
    'CENTER_OF_MASS_Z': 'center_of_mass_z'
}
