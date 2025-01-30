import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

app = dash.Dash()
app.layout = html.Div(children=[
    dcc.Graph(
        id='example',
        figure={
            'data': [{'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'}]
        }
    )
])

if __name__ == "__main__":
    app.run_server(debug=True)