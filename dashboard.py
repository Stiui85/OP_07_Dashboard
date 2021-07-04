import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import dash_auth

# Themes: https://www.bootstrapcdn.com/bootswatch
# className: https://hackerthemes.com/bootstrap-cheatsheet/

USERNAME_PASSWORD_PAIRS = {'KingAverage': '1234'}

# Imports
data_test = pd.read_csv("data_test.csv").drop("Unnamed: 0", axis=1)

df_plot = pd.read_csv("df_plot.csv")
X = df_plot.drop(["SK_ID_CURR", "TARGET", "Unnamed: 0"], axis=1)

# Retrieve categorical columns
data_numeric_columns = pickle.load(open("data_numeric_columns.sav", 'rb'))
data_non_numeric_columns = pickle.load(open("data_non_numeric_columns.sav", 'rb'))

df_plot_2 = pd.read_csv("df_plot_2.csv")
df_plot_2 = df_plot_2.drop("Unnamed: 0", axis=1)

xgb_grid = pickle.load(open("xgb_grid.sav", 'rb'))
model = xgb_grid.best_estimator_

# Separating numerical and categorical columns
X_num_columns = pickle.load(open("X_num_columns.sav", 'rb'))
X_cat_columns = pickle.load(open("X_cat_columns.sav", 'rb'))

# Retrieving original categorical columns
cat_columns = list(set([x.rpartition('_')[0] for x in X_cat_columns]))

user_ids = [int(x) for x in data_test["SK_ID_CURR"].values]
users = [{"label": x, "value": x} for x in user_ids]

feature_imp = model.feature_importances_
features = [{"label": x, "value": x} for x in X_num_columns]
features_cat = [{"label": x, "value": x} for x in cat_columns]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE],
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}])

auth = dash_auth.BasicAuth(app, USERNAME_PASSWORD_PAIRS)
server = app.server

app.layout = dbc.Container([
    html.Br(),
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                html.H3('Dashboard  by  KingA', style={'fontSize': 40, 'TextAlign': 'center'})
            ], justify='center')
        ], className='d-flex flex-wrap align-content-center justify-content-center')
    ]),
    html.Br(),
    dbc.CardDeck([
        dbc.Card([
            dbc.CardBody([
                dcc.Dropdown(
                    id='dropdown-user',
                    value="",
                    options=users,
                    style={'fontSize': 14}
                )
            ])
        ]),
        dbc.Card([
            dbc.CardBody([
                dcc.Slider(id='score_now', value=0.605, min=0, max=1, step=0.001,
                           marks={0: {'label': "Likely to pay"}, 1: {'label': "Unlikely to pay"},
                                  0.605: {'label': "Threshold"}}, tooltip={'always_visible': True})
            ])
        ]),
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    html.H3(id='score_result_now', style={'fontSize': 16, 'TextAlign': 'center'})
                ], justify='center')
            ], className='d-flex flex-wrap align-content-center justify-content-center')
        ])
    ]),
    html.Br(),
    dbc.CardDeck([
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    html.H4('Score above = current score / Score right = updated score',
                            style={'fontSize': 16, 'TextAlign': 'center'})
                ], justify='center')
            ], className='d-flex flex-wrap align-content-center justify-content-center')
        ]),
        dbc.Card([
            dbc.CardBody([
                dcc.Slider(id='score_after', value=0.605, min=0, max=1, step=0.001,
                           marks={0: {'label': "Likely to pay"}, 1: {'label': "Unlikely to pay"},
                                  0.605: {'label': "Threshold"}}, tooltip={'always_visible': True})
            ])
        ]),
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    html.H4(id='score_result_after', style={'fontSize': 20, 'TextAlign': 'center'})
                ], justify='center')
            ], className='d-flex flex-wrap align-content-center justify-content-center')
        ])
    ]),
    html.Br(),
    dbc.CardDeck([
        dbc.Card([
            dbc.CardBody([
                html.H5('Sub-select only the following number of numerical features:'),
                dcc.Input(id='limit_features', placeholder=str(len(X_num_columns)), type="number"),
                html.Br(),
                html.Br(),
                dcc.RadioItems(id='radio_features',
                               options=[
                                   {'label': 'Alphabetical', 'value': 'Alphabetical'},
                                   {'label': 'Importance', 'value': 'Importance'}
                               ],
                               value='Alphabetical',
                               labelStyle={'margin-left': '10px'}),
                dcc.Dropdown(
                    id='dropdown-feature',
                    options=features,
                    multi=True,
                    placeholder="Select a feature",
                    value=[],
                    style={'fontSize': 14}
                ),
                html.Br(),
                html.H5(id='feature_01', style={'fontSize': 16}),
                html.Div(dcc.Slider(id='feature_01_value', min=0, max=1, step=0.1, value=0.5,
                                    marks={0: {'label': '0'}, 1: {'label': '1'}})),
                html.Br(),
                html.H5(id='feature_02', style={'fontSize': 16}),
                html.Div(dcc.Slider(id='feature_02_value', min=0, max=1, step=0.1, value=0.5,
                                    marks={0: {'label': '0'}, 1: {'label': '1'}})),
                html.Br(),
                html.H5(id='feature_03', style={'fontSize': 16}),
                html.Div(dcc.Slider(id='feature_03_value', min=0, max=1, step=0.1, value=0.5,
                                    marks={0: {'label': '0'}, 1: {'label': '1'}}))
            ])
        ]),
        dbc.Card([
            dbc.CardBody([
                html.H5('The following are categorical features:'),
                dcc.Dropdown(
                    id='dropdown-feature_cat',
                    options=features_cat,
                    multi=True,
                    placeholder="Select a feature",
                    value=[],
                    style={'fontSize': 14}
                ),
                html.Br(),
                html.H5(id='feature_04', style={'fontSize': 16}),
                dcc.Dropdown(
                    id='feature_04_value',
                    options=[],
                    style={'fontSize': 14}
                ),
                html.Br(),
                html.H5(id='feature_05', style={'fontSize': 16}),
                dcc.Dropdown(
                    id='feature_05_value',
                    options=[],
                    style={'fontSize': 14}
                ),
                html.Br(),
                html.H5(id='feature_06', style={'fontSize': 16}),
                dcc.Dropdown(
                    id='feature_06_value',
                    options=[],
                    style={'fontSize': 14}
                )
            ])
        ])
    ]),
    html.Br(),
    dbc.CardDeck([
        dbc.Card([
            dbc.CardBody(
                dcc.Graph(
                    id='num_features_graph',
                    figure=px.box()
                )
            )
        ]),
        dbc.Card([
            dbc.CardBody(
                dcc.Graph(
                    id='cat_features_graph',
                    figure=px.box()
                )
            )
        ])
    ]),
    html.Br()
], fluid=True)


@app.callback(
    Output('score_now', 'value'),
    [Input('dropdown-user', 'value')])
def output(user):
    if user != "":
        probs = model.predict_proba(data_test[data_test["SK_ID_CURR"] == user].drop(["SK_ID_CURR", "TARGET"], axis=1))
        return round(probs[0][1], 3)
    else:
        return 0.605


@app.callback(
    Output('score_result_now', 'children'),
    [Input('dropdown-user', 'value')])
def output(user):
    if user != "":
        probs = model.predict_proba(data_test[data_test["SK_ID_CURR"] == user].drop(["SK_ID_CURR", "TARGET"], axis=1))
        if probs[0][1] > 0.605:
            return "The customer is likely NOT to repay the loan"
        else:
            return "The customer is likely to repay the loan"
    else:
        return "Select User for outcome"


@app.callback(
    Output('dropdown-feature', 'options'),
    [Input('radio_features', 'value'), Input('limit_features', 'value')])
def output(radio, number):
    if number is None or number > len(X_num_columns) or number < 1:
        number = len(X_num_columns)
    feature_df = pd.DataFrame({'label': X.columns, 'imp': feature_imp}).sort_values('imp', ascending=False)
    feature_df = feature_df[feature_df['label'].isin(X_num_columns)]
    features_num = feature_df.iloc[:round(number), 0].tolist()
    options = [{"label": x, "value": x} for x in features_num]
    if radio == 'Alphabetical':
        options = [{"label": x, "value": x} for x in sorted([x['label'] for x in options])]
    if radio == 'Importance':
        feature_df = pd.DataFrame({'label': X.columns, 'imp': feature_imp}).sort_values('imp', ascending=False)
        feature_df = feature_df[feature_df['label'].isin(X_num_columns)]
        features_num = feature_df.iloc[:round(len(options)), 0].tolist()
        options = [{"label": x, "value": x} for x in features_num]
    return options


@app.callback(
    Output('feature_01', 'children'),
    [Input('dropdown-feature', 'value')])
def output(feature):
    if len(feature) > 0:
        return feature[0]
    else:
        return 'You can select a first feature'


@app.callback(
    [Output('feature_01_value', 'min'), Output('feature_01_value', 'max'), Output('feature_01_value', 'step'),
     Output('feature_01_value', 'value'), Output('feature_01_value', 'marks')],
    [Input('dropdown-user', 'value'), Input('dropdown-feature', 'value')])
def output(user, feature):
    if len(feature) > 0:
        # diff = data_test[feature[0]].max() - data_test[feature[0]].min()
        res_min = data_test[feature[0]].min()
        res_max = data_test[feature[0]].max()
        # res_min = data_test[feature[0]].min() - diff/2
        # res_max = data_test[feature[0]].max() + diff/2
        res_step = (res_max - res_min) / 1000
        res_value = data_test[data_test["SK_ID_CURR"] == user][feature[0]].values[0]
        res_marks = {round(res_min): {'label': str(round(res_min))},
                     round(res_max): {'label': str(round(res_max))},
                     round(res_value): {'label': str(round(res_value))}}
        return res_min, res_max, res_step, res_value, res_marks
    else:
        return 0, 1, 0.1, 0.5, {0: {'label': '0'}, 1: {'label': '1'}}


@app.callback(
    Output('feature_02', 'children'),
    [Input('dropdown-feature', 'value')])
def output(feature):
    if len(feature) > 1:
        return feature[1]
    else:
        return 'You can select a second feature'


@app.callback(
    [Output('feature_02_value', 'min'), Output('feature_02_value', 'max'), Output('feature_02_value', 'step'),
     Output('feature_02_value', 'value'), Output('feature_02_value', 'marks')],
    [Input('dropdown-user', 'value'), Input('dropdown-feature', 'value')])
def output(user, feature):
    if len(feature) > 1:
        # diff = data_test[feature[1]].max() - data_test[feature[1]].min()
        res_min = data_test[feature[1]].min()
        res_max = data_test[feature[1]].max()
        # res_min = data_test[feature[1]].min() - diff/2
        # res_max = data_test[feature[1]].max() + diff/2
        res_step = (res_max - res_min) / 1000
        res_value = data_test[data_test["SK_ID_CURR"] == user][feature[1]].values[0]
        res_marks = {round(res_min): {'label': str(round(res_min))},
                     round(res_max): {'label': str(round(res_max))},
                     round(res_value): {'label': str(round(res_value))}}
        return res_min, res_max, res_step, res_value, res_marks
    else:
        return 0, 1, 0.1, 0.5, {0: {'label': '0'}, 1: {'label': '1'}}


@app.callback(
    Output('feature_03', 'children'),
    [Input('dropdown-feature', 'value')])
def output(feature):
    if len(feature) > 2:
        return feature[2]
    else:
        return 'You can select a third feature'


@app.callback(
    [Output('feature_03_value', 'min'), Output('feature_03_value', 'max'), Output('feature_03_value', 'step'),
     Output('feature_03_value', 'value'), Output('feature_03_value', 'marks')],
    [Input('dropdown-user', 'value'), Input('dropdown-feature', 'value')])
def output(user, feature):
    if len(feature) > 2:
        # diff = data_test[feature[2]].max() - data_test[feature[2]].min()
        res_min = data_test[feature[2]].min()
        res_max = data_test[feature[2]].max()
        # res_min = data_test[feature[2]].min() - diff/2
        # res_max = data_test[feature[2]].max() + diff/2
        res_step = (res_max - res_min) / 1000
        res_value = data_test[data_test["SK_ID_CURR"] == user][feature[2]].values[0]
        res_marks = {round(res_min): {'label': str(round(res_min))},
                     round(res_max): {'label': str(round(res_max))},
                     round(res_value): {'label': str(round(res_value))}}
        return res_min, res_max, res_step, res_value, res_marks
    else:
        return 0, 1, 0.1, 0.5, {0: {'label': '0'}, 1: {'label': '1'}}


@app.callback(
    Output('dropdown-feature_cat', 'options'),
    [Input('dropdown-user', 'value')])
def output(user):
    if user != "":
        feat_list = []
        for feat in X_cat_columns:
            if data_test[data_test["SK_ID_CURR"] == user][feat].values[0] == 1:
                feat_list.append(feat)
        feat_list = list(set([x.rpartition('_')[0] for x in feat_list]))
        feat_list = sorted(feat_list)
        options = [{'label': x, 'value': x} for x in feat_list]
        return options
    else:
        return []


@app.callback(
    Output('feature_04', 'children'),
    [Input('dropdown-feature_cat', 'value')])
def output(feature):
    if len(feature) > 0:
        return feature[0]
    else:
        return 'You can select a first feature'


@app.callback(
    Output('feature_04_value', 'value'),
    [Input('dropdown-user', 'value'), Input('dropdown-feature_cat', 'value')])
def output(user, feature):
    if len(feature) > 0:
        string_init = feature[0]
        feature_list = [x for x in X_cat_columns if x.startswith(string_init)]
        for feat in feature_list:
            if data_test[data_test["SK_ID_CURR"] == user][feat].values[0] == 1:
                return feat.rpartition('_')[2]
        return 'This feature cannot be changed'
    else:
        return ''


@app.callback(
    Output('feature_04_value', 'options'),
    [Input('dropdown-feature_cat', 'value')])
def output(feature):
    if len(feature) > 0:
        string_init = feature[0]
        feature_list = [x for x in X_cat_columns if x.startswith(string_init)]
        options = [x.rpartition('_')[2] for x in feature_list]
        return [{'label': x, 'value': x} for x in options]
    else:
        return []


@app.callback(
    Output('feature_05', 'children'),
    [Input('dropdown-feature_cat', 'value')])
def output(feature):
    if len(feature) > 1:
        return feature[1]
    else:
        return 'You can select a second feature'


@app.callback(
    Output('feature_05_value', 'value'),
    [Input('dropdown-user', 'value'), Input('dropdown-feature_cat', 'value')])
def output(user, feature):
    if len(feature) > 1:
        string_init = feature[1]
        feature_list = [x for x in X_cat_columns if x.startswith(string_init)]
        for feat in feature_list:
            if data_test[data_test["SK_ID_CURR"] == user][feat].values[0] == 1:
                return feat.rpartition('_')[2]
        return 'This feature cannot be changed'
    else:
        return ''


@app.callback(
    Output('feature_05_value', 'options'),
    [Input('dropdown-feature_cat', 'value')])
def output(feature):
    if len(feature) > 1:
        string_init = feature[1]
        feature_list = [x for x in X_cat_columns if x.startswith(string_init)]
        options = [x.rpartition('_')[2] for x in feature_list]
        return [{'label': x, 'value': x} for x in options]
    else:
        return []


@app.callback(
    Output('feature_06', 'children'),
    [Input('dropdown-feature_cat', 'value')])
def output(feature):
    if len(feature) > 2:
        return feature[2]
    else:
        return 'You can select a third feature'


@app.callback(
    Output('feature_06_value', 'value'),
    [Input('dropdown-user', 'value'), Input('dropdown-feature_cat', 'value')])
def output(user, feature):
    if len(feature) > 2:
        string_init = feature[2]
        feature_list = [x for x in X_cat_columns if x.startswith(string_init)]
        for feat in feature_list:
            if data_test[data_test["SK_ID_CURR"] == user][feat].values[0] == 1:
                return feat.rpartition('_')[2]
        return 'This feature cannot be changed'
    else:
        return ''


@app.callback(
    Output('feature_06_value', 'options'),
    [Input('dropdown-feature_cat', 'value')])
def output(feature):
    if len(feature) > 2:
        string_init = feature[2]
        feature_list = [x for x in X_cat_columns if x.startswith(string_init)]
        options = [x.rpartition('_')[2] for x in feature_list]
        return [{'label': x, 'value': x} for x in options]
    else:
        return []


@app.callback(
    Output('num_features_graph', 'figure'),
    [Input('feature_01', 'children'), Input('feature_02', 'children'), Input('feature_03', 'children')])
def output(feature_01, feature_02, feature_03):
    if feature_01 != 'You can select a first feature':
        features_list = [feature_01]
        if feature_02 != 'You can select a second feature':
            features_list.append(feature_02)
            if feature_03 != 'You can select a third feature':
                features_list.append(feature_03)
        temp = df_plot[features_list]
        feat_axes = ["y", "y2", "y3"]
        fig = go.Figure()
        for i in range(len(features_list)):
            fig.add_trace(go.Box(y=temp[features_list[i]], yaxis=feat_axes[i], name=features_list[i]))
        if feature_02 == "You can select a second feature" and feature_03 == "You can select a third feature":
            fig.update_layout(yaxis=dict(title=feature_01, titlefont=dict(color="#1f77b4"),
                                         tickfont=dict(color="#1f77b4")),
                              margin=dict(l=20, r=20, t=20, b=20))
        if feature_02 != 'You can select a second feature' and feature_03 == "You can select a third feature":
            fig.update_layout(yaxis=dict(title=feature_01, titlefont=dict(color="#1f77b4"),
                                         tickfont=dict(color="#1f77b4")),
                              yaxis2=dict(title=feature_02, titlefont=dict(color="#d62728"),
                                          tickfont=dict(color="#d62728"), anchor="x", overlaying="y",
                                          side="right"),
                              margin=dict(l=20, r=20, t=20, b=20))
        if feature_03 != 'You can select a third feature':
            fig.update_layout(xaxis=dict(domain=[0.25, 1.0]),
                              yaxis=dict(title=feature_01, titlefont=dict(color="#1f77b4"),
                                         tickfont=dict(color="#1f77b4")),
                              yaxis2=dict(title=feature_02, titlefont=dict(color="#d62728"),
                                          tickfont=dict(color="#d62728"), anchor="free", overlaying="y",
                                          side="right", position=0.),
                              yaxis3=dict(title=feature_03, titlefont=dict(color="#9467bd"),
                                          tickfont=dict(color="#9467bd"), anchor="x", overlaying="y",
                                          side="right"),
                              margin=dict(l=20, r=20, t=20, b=20))
        return fig
    else:
        fig = px.box()
        return fig


@app.callback(
    Output('cat_features_graph', 'figure'),
    [Input('feature_04', 'children'), Input('feature_05', 'children'), Input('feature_06', 'children')])
def output(feature_04, feature_05, feature_06):
    if feature_04 != 'You can select a first feature':
        features_list = [feature_04]
        if feature_05 != 'You can select a second feature':
            features_list.append(feature_05)
            if feature_06 != 'You can select a third feature':
                features_list.append(feature_06)
        temp = df_plot_2[features_list]
        fig = go.Figure()
        for i in range(len(features_list)):
            data_col = temp[features_list[i]] + str(i)
            fig.add_trace(go.Bar(y=data_col.value_counts(), x=data_col.unique(), name=features_list[i]))
        return fig
    else:
        fig = px.box()
        return fig


@app.callback(
    Output('score_after', 'value'),
    [Input('dropdown-user', 'value'), Input('feature_01_value', 'value'), Input('feature_02_value', 'value'),
     Input('feature_03_value', 'value'), Input('feature_04_value', 'value'), Input('feature_05_value', 'value'),
     Input('feature_06_value', 'value'), Input('feature_01', 'children'), Input('feature_02', 'children'),
     Input('feature_03', 'children'), Input('feature_04', 'children'), Input('feature_05', 'children'),
     Input('feature_06', 'children')])
def output(user, feature_01, feature_02, feature_03, feature_04, feature_05, feature_06, f01_name, f02_name, f03_name,
           f04_name, f05_name, f06_name):
    if user != "":
        data_test_new = data_test[data_test["SK_ID_CURR"] == user].drop(["SK_ID_CURR", "TARGET"], axis=1)
        if f01_name != "You can select a first feature" and feature_01 is not None:
            data_test_new[f01_name] = feature_01
        if f02_name != "You can select a second feature" and feature_02 is not None:
            data_test_new[f02_name] = feature_02
        if f03_name != "You can select a third feature" and feature_03 is not None:
            data_test_new[f03_name] = feature_03
        if f04_name != "You can select a first feature" and feature_04 is not None:
            list_features = [x for x in X_cat_columns if x.startswith(f04_name)]
            for feat in list_features:
                data_test_new[feat] = 0
            feature_name = f04_name + "_" + feature_04
            data_test_new[feature_name] = 1
        if f05_name != "You can select a second feature" and feature_05 is not None:
            list_features = [x for x in X_cat_columns if x.startswith(f05_name)]
            for feat in list_features:
                data_test_new[feat] = 0
            feature_name = f05_name + "_" + feature_05
            data_test_new[feature_name] = 1
        if f06_name != "You can select a third feature" and feature_06 is not None:
            list_features = [x for x in X_cat_columns if x.startswith(f06_name)]
            for feat in list_features:
                data_test_new[feat] = 0
            feature_name = f06_name + "_" + feature_06
            data_test_new[feature_name] = 1
        probs = model.predict_proba(data_test_new)
        return round(probs[0][1], 3)
    else:
        return 0.605


@app.callback(
    Output('score_result_after', 'children'),
    [Input('dropdown-user', 'value'), Input('score_after', 'value')])
def output(user, score):
    if user != "":
        if score > 0.605:
            return "The customer is likely NOT to repay the loan"
        else:
            return "The customer is likely to repay the loan"
    else:
        return "Select User for outcome"


if __name__ == '__main__':
    app.run_server(debug=True)
