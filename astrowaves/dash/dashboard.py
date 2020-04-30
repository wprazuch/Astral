import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

path = r'C:\Users\Wojtek\Documents\Doktorat\AstrocyteCalciumWaveDetector\debug\timespace.npy'

timespace = np.load(path)


# app.layout = html.Div([
#     dcc.Graph(
#         id='life-exp-vs-gdp',
#         figure={
#             'data': [
#                 dict(
#                     x=df[df['continent'] == i]['gdp per capita'],
#                     y=df[df['continent'] == i]['life expectancy'],
#                     text=df[df['continent'] == i]['country'],
#                     mode='markers',
#                     opacity=0.7,
#                     marker={
#                         'size': 15,
#                         'line': {'width': 0.5, 'color': 'white'}
#                     },
#                     name=i
#                 ) for i in df.continent.unique()
#             ],
#             'layout': dict(
#                 xaxis={'type': 'log', 'title': 'GDP Per Capita'},
#                 yaxis={'title': 'Life Expectancy'},
#                 margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
#                 legend={'x': 0, 'y': 1},
#                 hovermode='closest'
#             )
#         }
#     )
# ])
import plotly.express as px

active_frame = timespace[:,:,0]

fig = px.imshow(active_frame, color_continuous_scale='gray')

app.layout = html.Div([
    dcc.Graph(
        id='crossfilter-indicator-scatter',
        figure=fig
        ),

    dcc.Slider(
        id='crossfilter-year--slider',
        min=0,
        max=1200,
        marks={i: 'Slice {}'.format(i) if i == 1 else str(i) for i in range(0, 1201, 50)},
        value=5,
    )
])

@app.callback(
    dash.dependencies.Output('crossfilter-indicator-scatter', 'figure'),
    [
     dash.dependencies.Input('crossfilter-year--slider', 'value')
    ])
def update_graph(year_value):
    active_frame = timespace[:,:,year_value]

    return px.imshow(active_frame, color_continuous_scale='gray')





if __name__ == '__main__':
    app.run_server(debug=True)