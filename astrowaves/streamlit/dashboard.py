import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st

path = r'C:\Users\Wojtek\Documents\Doktorat\AstrocyteCalciumWaveDetector\debug\timespace.npy'
rel_path = r'C:\Users\Wojtek\Documents\Doktorat\AstrocyteCalciumWaveDetector\debug\segmentation_relative.h5'
abs_path = r'C:\Users\Wojtek\Documents\Doktorat\AstrocyteCalciumWaveDetector\debug\segmentation_absolute.h5'
dims_path = r'C:\Users\Wojtek\Documents\Doktorat\AstrocyteCalciumWaveDetector\debug\segmentation_dims.h5'

timespace = np.load(path)
rel = pd.read_hdf(rel_path)
abss = pd.read_hdf(abs_path)
dims = pd.read_hdf(dims_path)
dims = dims.astype('int')

t_dims = timespace.shape

x_range = st.slider("X range", 1, t_dims[1], (1, t_dims[1]))
y_range = st.slider("Y range", 1, t_dims[0], (1, t_dims[0]))
z_range = st.slider("Z range", 1, t_dims[2], (1, t_dims[2]))

dims = dims.loc[(dims['center_x'] > x_range[0]) & (dims['center_x'] < x_range[1])]
dims = dims.loc[(dims['center_y'] > y_range[0]) & (dims['center_y'] < y_range[1])]
dims = dims.loc[(dims['center_z'] > z_range[0]) & (dims['center_z'] < z_range[1])]

st.write(dims)

which_slice = st.slider("Select a slice of a timespace: ", min_value=1, max_value=1200)
active_frame = timespace[:, :, which_slice-1]
fig = px.imshow(active_frame, color_continuous_scale='gray')
points = dims.loc[dims['center_z'] == which_slice]

# fig.add_trace(px.scatter(points,
#                          x=points['center_x'],
#                          y=points['center_y'],
#                          hover_data=['id']))

# fig.add_trace(go.Scatter(points,
#                          x=points['center_x'],
#                          y=points['center_y'],
#                          marker=dict(color='red', size=4),
#                          hover_data=['id']))

# fig2 = px.scatter(points, x="center_x", y="center_y",
#                   hover_name="id")

fig2 = go.Scatter(x=points['center_x'], y=points['center_y'], mode='markers', text=points['id'])

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
            colorscale='hot',   # choose a colorscale
            opacity=1
        )
    )])
    return fig


shape_id = st.slider("Select shape id: ", min_value=int(rel['id'].min()), max_value=int(rel['id'].max()))
shape1 = rel.loc[rel['id'] == shape_id]
fig3d = scatter_3d(shape1)
st.write(fig3d)
