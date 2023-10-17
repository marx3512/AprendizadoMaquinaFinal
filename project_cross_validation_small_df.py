# %% Aqui esta usando a coluna dos géneros dos filmes

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from IPython.display import display
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error

data = pd.read_csv('dataset_small/ratings.csv')

X = data.drop(columns=['rating', 'timestamp'])
y = data['rating']

k_folds = [5, 10]

scalers = {
    'without_scaler': None,
    'min_max': MinMaxScaler(),
    'z_score': StandardScaler()
}

regressors = {
    'knn': KNeighborsRegressor(n_neighbors=15),
    'decicion_tree': DecisionTreeRegressor(),
    'linear_regression': LinearRegression(),
    'random_forest': RandomForestRegressor(),
    'neural_network_mlp': MLPRegressor()
}

scoring = {
    'mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False),
    'mean_absolute_error': make_scorer(mean_absolute_error, greater_is_better=False)
}
# %% KNN

n_neighbors =  [5, 9, 13, 15]
weights = ['distance', 'uniform']

dataframe_mse = {}
dataframe_mae = {}

for k in k_folds:
    for num in n_neighbors:
        for weight in weights:
            knnRegressor = KNeighborsRegressor(num, weights=weight)

            for scaler in scalers:
                if(scalers[scaler] == None):
                    pipe = Pipeline([('regressor', knnRegressor)])
                else:
                    pipe = Pipeline([(scaler, scalers[scaler]), ('regressor', knnRegressor)])

                results = cross_validate(pipe, X, y, cv=k, scoring=scoring)

                mse_scores = np.mean(np.abs(results['test_mean_squared_error']))
                mae_scores = np.mean(np.abs(results['test_mean_absolute_error']))

                key = f'{k}_fold-{num}_neighbors-{weight}'

                if key in dataframe_mse:
                    dataframe_mse[key].append(mse_scores)
                    dataframe_mae[key].append(mae_scores)
                else:
                    dataframe_mse[key] = [mse_scores]
                    dataframe_mae[key] = [mae_scores]

mse_df = pd.DataFrame.from_dict(dataframe_mse, orient='index', columns=['without-scaler', 'min-max', 'z-score'])
mae_df = pd.DataFrame.from_dict(dataframe_mae, orient='index', columns=['without-scaler', 'min-max', 'z-score'])

display('MSE', mse_df)
display('MAE', mae_df)

# %% Comparação entre os métodos

dataframe_mse = {}
dataframe_mae = {}

for k in k_folds:
    for regressor in regressors:
        for scaler in scalers:
            if(scalers[scaler] == None):
                pipe = Pipeline([('regressor', regressors[regressor])])
            else:
                pipe = Pipeline([(scaler, scalers[scaler]), ('regressor', regressors[regressor])])

            results = cross_validate(pipe, X, y, cv=k, scoring=scoring)

            mse_scores = np.mean(np.abs(results['test_mean_squared_error']))
            mae_scores = np.mean(np.abs(results['test_mean_absolute_error']))

            key = f'{k}_fold-{regressor}'

            if key in dataframe_mse:
                dataframe_mse[key].append(mse_scores)
                dataframe_mae[key].append(mae_scores)
            else:
                dataframe_mse[key] = [mse_scores]
                dataframe_mae[key] = [mae_scores]

mse_df = pd.DataFrame.from_dict(dataframe_mse, orient='index', columns=['without-scaler', 'min-max', 'z-score'])
mae_df = pd.DataFrame.from_dict(dataframe_mae, orient='index', columns=['without-scaler', 'min-max', 'z-score'])

display('MSE', mse_df)
display('MAE', mae_df)
