#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Built-in/Generic Imports
import math
import copy

# Libs
import pandas as pd
import numpy as np

import dash
import dash_html_components as html
import dash_core_components as dcc

import plotly.express as px

# Own modules
import utils
from utils import NoneType;


def drawMatrix(df, value1, value2, column):

    colours = df[column].fillna(0);

    size_scale = 50;
    size = colours.abs();

    fig = px.scatter(
        df,
        x='x',#x.map(x_to_num), # Use mapping for x
        y='y',#y.map(y_to_num), # Use mapping for y
        size=size * size_scale, # Vector of square sizes, proportional to size parameter
        color=colours, # Vector of square color values, mapped to color palette
        color_continuous_scale=px.colors.diverging.Picnic,
        # range_color=[-1,1],
        labels={'x': value1, 'y': value2},
        width=800, height=800,
    );

    fig.update_traces(marker=dict(
                                  line=dict(width=2,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))

    return fig;


def getCorrelation(df, normalise_data):

    if normalise_data:
        field = "Normalised count"
    else:
        field = "Count"

    corr = [];
    x_val = [];
    dates = [];
    lag = 0;

    if len(df["data"].unique()) == 2:
        field1 = df["data"].unique()[0];
        field2 = df["data"].unique()[1];

        dates = [];
        series1 = [];
        series2 = [];

        for date in df["Date"].unique():

            test1 = df["Date"] == date;

            if df[test1][field].count() == 2:

                test2 = df["data"] == field1;
                test3 = df["data"] == field2;

                dates.append(date);
                series1.append(df[test1 & test2][field].values[0]);
                series2.append(df[test1 & test3][field].values[0]);

        if len(dates) % 2 == 0:
            x_val = range(-math.floor(len(dates)/2), math.floor(len(dates)/2), 1);
        else:
            x_val = range(-math.floor(len(dates)/2), math.floor(len(dates)/2) + 1, 1);

        corr = utils.correlateTimeSeries(series1, series2);
        lag = x_val[np.argmax(corr, axis=0)];

    return corr, x_val, lag, dates;


def drawCrossCorr(df, normalise_data):

    if normalise_data:
        field = "Normalised count"
    else:
        field = "Count"

    corr, x_val, lag, dates = getCorrelation(df, normalise_data);

    if len(dates):
        return px.scatter(x=x_val, y=corr, labels={'x': "Lab (in # of days)", 'y': "Cross correlation"});

    return {'data': []};


def drawTimeLine(df, normalise_data, lag):


    series1 = [];
    series2 = [];

    if normalise_data:
        field = "Normalised count"
    else:
        field = "Count"

    # Add the lag
    date=df["Date"];

    # print("LAG=", lag, "   ", type(lag));


    if lag != 0:
        if len(df["data"].unique()) == 2:
            date = [];
            field1 = df["data"].unique()[0];
            field2 = df["data"].unique()[1];

            df["corrected_date"] = copy.deepcopy(df["Date"].values);

            test1 = df["data"] == field1;
            test2 = df["data"] == field2;

            # print(test2)

            # df[test2]['corrected_date'] = df[test2]['Date'] + pd.DateOffset(days=lag);
            # df[test2]['corrected_date'] += pd.DateOffset(days=lag);

            df['corrected_date'] = df['Date'].copy();

            for t, row in zip(test2, df['corrected_date']):
                if t:
                    row += pd.DateOffset(days=lag);
                date.append(row);
                # print(t, row);
            # print(df.shape)
            # print(test2)
            # print(df[test2]['corrected_date'])
            # df[test2]['corrected_date'] = df[test2]['Date'] + pd.DateOffset(days=lag);
            # print(df['corrected_date'])

            # date=df["corrected_date"];

            # for i in range(len(date)):
            #     print (i, date[i]);

            # for d1, d2, d3 in zip(df[test1]['Date'], df[test2]['Date'], df[test2]['corrected_date']):
            #     print(d1, d2, d3)

    return px.scatter(df, x=date, y=field, color='data');


external_stylesheets = ['app.css', 'https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets);

data_options = [];
filter_options = [];
correlation_options = [];

for file in utils.data_files:
    short_filename = file[:len(file)-4];
    data_options.append({'label': short_filename, 'value': short_filename});

for filter in utils.window_set:
    filter_options.append({'label': filter, 'value': filter});

for correlation in utils.similarity_metrics_set:
    correlation_options.append({'label': correlation, 'value': correlation});

data_file_dropdown1 = dcc.Dropdown(
        id='data-file-dropdown1',
        options=data_options,
        value=data_options[1]['value'],
        searchable=False,
        clearable=False,
        # placeholder="Select a data file",
    );

data_file_dropdown2 = dcc.Dropdown(
        id='data-file-dropdown2',
        options=data_options,
        value=data_options[1]['value'],
        searchable=False,
        clearable=False,
        # placeholder="Select a data file",
    );

window_dropdown = dcc.Dropdown(
        id='window-dropdown',
        options=filter_options,
        value=filter_options[0]['value'],
        searchable=False,
        clearable=False,
        # placeholder="Select a filter",
    );

correlation_dropdown = dcc.Dropdown(
        id='correlation_dropdown',
        options=correlation_options,
        value=correlation_options[0]['value'],
        searchable=False,
        clearable=False,
        # placeholder="Select a filter",
    );



normalise_data_checkbox = dcc.Checklist(
    options=[
        {'label': 'zero-mean and unit-variance normalisation', 'value': 'normalise_data'},
    ],
    value=['normalise_data'],
    id='normalise_data_checkbox',
)

app.layout = html.Div([
    html.H1(children='Zero mean Normalized Cross-Correlation of Time Series'),

    data_file_dropdown1,
    data_file_dropdown2,
    window_dropdown,
    correlation_dropdown,
    html.Div(id='dd-output-container1'),
    html.Div(id='dd-output-container2'),
    html.Div(id='dd-output-container3'),
    normalise_data_checkbox,

    html.Div(
        [
            # html.Div([
            #     html.H3(children='Correlation matrix'),
                dcc.Graph(id='corr_matrix', className="six columns"),
            # ]),

            # html.Div([
                # html.H3(children='Uncorrected time series'),
                dcc.Graph(id='time_line', className="six columns"),
            # ]),
            #
            # html.Div([
                # html.H3(children='Corrected time series'),
                dcc.Graph(id='corrected_time_line', className="six columns"),
            # ]),
            #
            # html.Div([
                # html.H3(children='Cross correlation'),
                dcc.Graph(id='cross_corr', className="six columns"),
            # ],
            # className="two columns"),
        ],
        className="row",
    ),
]);




@app.callback(
    dash.dependencies.Output('corr_matrix', 'figure'),
    [dash.dependencies.Input('data-file-dropdown1', 'value'),
    dash.dependencies.Input('data-file-dropdown2', 'value'),
    dash.dependencies.Input('window-dropdown', 'value'),
    dash.dependencies.Input('correlation_dropdown', 'value')])

def update_output(value1, value2, value3, correlation):
    df = utils.getSimilarityMetrics(value1, value2, value3)
    return drawMatrix(df, value1, value2, correlation);


# @app.callback(
#     dash.dependencies.Output('dd-output-container1', 'children'),
#     [dash.dependencies.Input('data-file-dropdown1', 'value')])
#
# def update_output(value):
#     getData()
#     return 'You have selected "{}"'.format(value)
#

@app.callback(
    dash.dependencies.Output('time_line', 'figure'),
    [dash.dependencies.Input('data-file-dropdown1', 'value'),
    dash.dependencies.Input('data-file-dropdown2', 'value'),
    dash.dependencies.Input('window-dropdown', 'value'),
    dash.dependencies.Input('corr_matrix', 'clickData'),
    dash.dependencies.Input('normalise_data_checkbox', 'value')])

def update_output(value1, value2, window, clickData, normalise_data):

    if not isinstance(clickData, NoneType):
        df1 = utils.getTimeSeries(value1, clickData['points'][0]['x'], window);
        df2 = utils.getTimeSeries(value2, clickData['points'][0]['y'], window);

        df = df1.append(df2);

        return drawTimeLine(df, normalise_data, 0);

    return {'data': []};


@app.callback(
    dash.dependencies.Output('cross_corr', 'figure'),
    [dash.dependencies.Input('data-file-dropdown1', 'value'),
    dash.dependencies.Input('data-file-dropdown2', 'value'),
    dash.dependencies.Input('window-dropdown', 'value'),
    dash.dependencies.Input('corr_matrix', 'clickData'),
    dash.dependencies.Input('normalise_data_checkbox', 'value')])

def update_output(value1, value2, window, clickData, normalise_data):

    if not isinstance(clickData, NoneType):
        df1 = utils.getTimeSeries(value1, clickData['points'][0]['x'], window);
        df2 = utils.getTimeSeries(value2, clickData['points'][0]['y'], window);

        df = df1.append(df2);

        return drawCrossCorr(df, normalise_data);

    return {'data': []};

@app.callback(
    dash.dependencies.Output('corrected_time_line', 'figure'),
    [dash.dependencies.Input('data-file-dropdown1', 'value'),
    dash.dependencies.Input('data-file-dropdown2', 'value'),
    dash.dependencies.Input('window-dropdown', 'value'),
    dash.dependencies.Input('corr_matrix', 'clickData'),
    dash.dependencies.Input('normalise_data_checkbox', 'value')])

def update_output(value1, value2, window, clickData, normalise_data):

    if not isinstance(clickData, NoneType):
        df1 = utils.getTimeSeries(value1, clickData['points'][0]['x'], window);
        df2 = utils.getTimeSeries(value2, clickData['points'][0]['y'], window);

        df = df1.append(df2);

        corr, x_val, lag, dates = getCorrelation(df, normalise_data);

        return drawTimeLine(df, normalise_data, lag);

    return {'data': []};


if __name__ == '__main__':
    app.run_server(debug=True);
