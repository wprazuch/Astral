import os

ROOT_DATA_DIR = "/app/data"


ABS_DF_COLUMNS = {"ID": "id", "Y": "y", "X": "x", "Z": "z", "COLOR": "color"}

REL_DF_COLUMNS = ABS_DF_COLUMNS.copy()


DIMENSIONS_DF_COLUMNS = {
    "ID": "id",
    "Y_MIN": "y_min",
    "Y_MAX": "y_max",
    "X_MIN": "x_min",
    "X_MAX": "x_max",
    "Z_MIN": "z_min",
    "Z_MAX": "z_max",
    "CENTER_Y": "center_y",
    "CENTER_X": "center_x",
    "CENTER_Z": "center_z",
    "CENTER_OF_MASS_X": "center_of_mass_x",
    "CENTER_OF_MASS_Y": "center_of_mass_y",
    "CENTER_OF_MASS_Z": "center_of_mass_z",
}

SINGLES_DF_COLUMNS = {
    "SHAPE_ID": "Calcium Wave ID",
    "MAX_X_SIZE": "Maximum X Size",
    "MAX_Y_SIZE": "Maximum Y Size",
    "MAX_Z_SIZE": "Maximum T Size",
    "MAX_XY_DIAMETER": "Maximum XY Diameter",
    "CIRCULARITY": "Circularity",
}

REPEATS_DF_COLUMNS = {
    "SHAPE_IDS": "Shape IDs",
    "NO_REPEATS": "Number of Repeats",
    "AVG_MAXIMUM_X": "Average Maximum X Size",
    "AVG_MAXIMUM_Y": "Average Maximum Y Size",
    "AVG_MAXIMUM_Z": "Average Maximum T Size",
    "AVG_MAXIMUM_XY": "Average Maximum XY Diameter",
    "MED_INT_REP_MIN_Z_DIST": "Median Inter Repeat Minimum T Distance",
    "MED_INT_REP_CENTER_DIST": "Median Inter Repeat Center Distance",
    "AVG_CIRCULARITY": "Average Circularity",
}
