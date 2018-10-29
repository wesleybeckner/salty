import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
import re
from math import log
import numpy as np

app = dash.Dash()

df = pd.read_csv("assets/vizapp.csv")


app.layout = html.Div([
    dcc.Graph(
        id='cpt-vs-density',
        figure={
            'data': [
                go.Scatter(
                    x=df[df['category'] == i]['Heat capacity at constant pressure, J/K/mol_mean'],
                    y=df[df['category'] == i]['Specific density, kg/m<SUP>3</SUP>_mean'],
                    text=df[df['category'] == i]['smiles-anion'],
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15+3*np.log(df[df['category'] == i]['count']),
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=i
                ) for i in df.category.unique()
            ],
            'layout': go.Layout(
                xaxis={'title': 'Heat Capacity'},
                yaxis={'title': 'Density'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},

                #legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
        }
    )
])

if __name__ == '__main__':
    app.run_server()
