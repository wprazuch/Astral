import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from radiomics.shape import RadiomicsShape
import SimpleITK as sitk

import streamlit as st
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', help="Specify path to meta data")


try:
    args = parser.parse_args()
except SystemExit as e:
    # This exception will be raised if --help or invalid command line arguments
    # are used. Currently streamlit prevents the program from exiting normally
    # so we have to do a hard exit.
    os._exit(e.code)

root_dir = args.path

dirs = tuple([file for file in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, file))])
print(dirs)
option = st.sidebar.selectbox('Select which sequence to display', dirs, index=0)

main_path = os.path.join(root_dir, option)


def plot_annotated_timespace(active_frame, dims):
    fig = px.imshow(active_frame, color_continuous_scale='inferno')
    points = dims.loc[dims['center_z'] == which_slice]
    fig2 = go.Scatter(x=points['center_x'], y=points['center_y'], mode='markers', text=points['id'], line=dict(
        color='white',
        width=2
    ))
    fig.add_trace(fig2)
    st.write(fig)


def scatter_3d(df):
    fig = go.Figure(data=[go.Scatter3d(
        x=df['x'],
        y=df['y'],
        z=df['z'],
        mode='markers',
        marker=dict(
            size=5,
            color=df['color'],                # set color to an array/list of desired values
            colorscale='inferno',   # choose a colorscale
            opacity=1
        )
    )])
    return fig


@st.cache(allow_output_mutation=True)
def load_data(main_path):
    timespace_path = os.path.join(main_path, 'timespace.npy')
    rel_path = os.path.join(main_path, 'segmentation_relative.h5')
    abs_path = os.path.join(main_path, 'segmentation_absolute.h5')
    dims_path = os.path.join(main_path, 'segmentation_dims.h5')
    waves_path = os.path.join(main_path, 'waves_morph.npy')

    timespace = np.load(timespace_path)
    waves = np.load(waves_path).astype('uint8')
    rel = pd.read_hdf(rel_path)
    abss = pd.read_hdf(abs_path)
    dims = pd.read_hdf(dims_path).astype('int')

    return timespace, waves, rel, abss, dims


def filter_dims_range(dims, x_range, y_range, z_range):
    dims_new = dims.loc[(dims['center_x'] > x_range[0]) & (dims['center_x'] < x_range[1])]
    dims_new = dims_new.loc[(dims['center_y'] > y_range[0]) & (dims['center_y'] < y_range[1])]
    dims_new = dims_new.loc[(dims['center_z'] > z_range[0]) & (dims['center_z'] < z_range[1])]

    return dims_new


timespace, waves, rel, abss, dims = load_data(main_path)


t_dims = timespace.shape
no_shapes = dims.shape[0]


x_range = st.sidebar.slider("X range", 1, t_dims[1], (1, t_dims[1]))
y_range = st.sidebar.slider("Y range", 1, t_dims[0], (1, t_dims[0]))
z_range = st.sidebar.slider("Z range", 1, t_dims[2], (1, t_dims[2]))


dims_new = filter_dims_range(dims, x_range, y_range, z_range)

if st.sidebar.button('Export CSV'):
    dims_new.to_csv(os.path.join(main_path, 'astral_data.csv'))


display_dims = dims_new.iloc[:, 1:]
st.subheader(f'Found {display_dims.shape[0]} instances.')
st.write(display_dims)


which_slice = st.number_input("Select a slice of a timespace: ", min_value=1, max_value=1200, step=1)

active_frame = timespace[:, :, which_slice - 1]

plot_annotated_timespace(active_frame, dims_new)


shape_id = st.number_input("Select id of the shape: ", min_value=1, max_value=no_shapes, step=1)
shape1 = rel.loc[rel['id'] == shape_id]
fig3d = scatter_3d(shape1)
st.write(fig3d)
