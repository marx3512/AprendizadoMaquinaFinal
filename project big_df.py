# %%

# Aqui esta usando a coluna de géneros dos filmes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ratings = pd.read_csv('dataset_small/ratings.csv')
movies = pd.read_csv('dataset_small/movies.csv')

merged = ratings.merge(movies[['movieId', 'genres']], on='movieId', how='left')

genres_encoded = merged['genres'].str.get_dummies(sep='|')
data = pd.concat([merged, genres_encoded], axis=1)
data.drop('genres', axis=1, inplace=True)

X = data.drop(columns=['rating', 'timestamp'])
y = data['rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

scalers = {
    'without_scaler': None,
    'min_max': MinMaxScaler(),
    'z-score': StandardScaler()
}

regressors = {
    'knn': KNeighborsRegressor(n_neighbors=15),
    'decicion_tree': DecisionTreeRegressor(),
    'linear_regression': LinearRegression(),
    'random_forest': RandomForestRegressor(),
    'neural_network_mlp': MLPRegressor()
}

# %%

# KNN

n_neighbors =  [5, 7, 9, 11, 13, 15]
weights = ['distance', 'uniform']

dataframe_mse = {}
dataframe_mae = {}

for num in n_neighbors:
    for weight in weights:
        for scaler in scalers:
            if(scalers[scaler] == None):
                pipe = Pipeline([('regressor', KNeighborsRegressor(num, weights=weight))])
            else:
                pipe = Pipeline([(scaler, scalers[scaler]), ('regressor', KNeighborsRegressor(num, weights=weight))])

            pipe.fit(X_train, y_train)
            y_predict = pipe.predict(X_test)

            mse = mean_squared_error(y_test, y_predict)
            mae = mean_absolute_error(y_test, y_predict)

            key = f'k_{num}-{weight}'

            if key in dataframe_mse:
                dataframe_mse[key].append(mse)
                dataframe_mae[key].append(mae)
            else:
                dataframe_mse[key] = [mse]
                dataframe_mae[key] = [mae]

            if num == 15:
                y_test_media = []
                y_predict_media = []
                tamanhoGrupo = 500
                for i in range(0, len(y_test), tamanhoGrupo):
                    y_predict_media.append(np.mean(y_predict[i:i + tamanhoGrupo]))
                    y_test_media.append(np.mean(y_test[i:i + tamanhoGrupo]))

                plt.scatter(y_predict_media, y_test_media)
                plt.xlabel("Predict")
                plt.ylabel("Test")
                plt.title(
                    f"Gráfico de Dispersão(Dividir em grupos de 500 e tirando a media, {weight.title()},{scaler.title()})")
                plt.show()

                plt.scatter(y_predict, y_test)
                plt.xlabel("Predict")
                plt.ylabel("Test")
                plt.title(
                    f"Gráfico de Dispersão(SEM Dividir em grupos de 500 e tirando a media, {weight.title()},{scaler.title()})")
                plt.show()

mse_df = pd.DataFrame.from_dict(dataframe_mse, orient='index', columns=['without-scaler', 'min-max', 'z-score'])
mae_df = pd.DataFrame.from_dict(dataframe_mae, orient='index', columns=['without-scaler', 'min-max', 'z-score'])

display('MSE', mse_df)
display('MAE', mae_df)

# %%

# Regressão Linear

dataframe_mse = {}
dataframe_mae = {}
dataframe_r2 = {}

for scaler in scalers:
    if(scalers[scaler] == None):
        pipe = Pipeline([('regressor', LinearRegression())])
    else:
        pipe = Pipeline([(scaler, scalers[scaler]), ('regressor', LinearRegression())])

    pipe.fit(X_train, y_train)
    y_predict = pipe.predict(X_test)

    mse = mean_squared_error(y_test, y_predict)
    mae = mean_absolute_error(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)

    key = f'metric'

    if key in dataframe_mse:
        dataframe_mse[key].append(mse)
        dataframe_mae[key].append(mae)
        dataframe_r2[key].append(r2)
    else:
        dataframe_mse[key] = [mse]
        dataframe_mae[key] = [mae]
        dataframe_r2[key] = [r2]

    y_test_media = []
    y_predict_media = []
    tamanhoGrupo = 500
    for i in range(0, len(y_test), tamanhoGrupo):
        y_predict_media.append(np.mean(y_predict[i:i + tamanhoGrupo]))
        y_test_media.append(np.mean(y_test[i:i + tamanhoGrupo]))

    plt.scatter(y_predict_media, y_test_media)
    plt.xlabel("Predict")
    plt.ylabel("Test")
    plt.title(
        f"Gráfico de Dispersão(Dividir em grupos de 500 e tirando a media, {scaler.title()})")
    plt.show()

    plt.scatter(y_predict, y_test)
    plt.xlabel("Predict")
    plt.ylabel("Test")
    plt.title(
        f"Gráfico de Dispersão(SEM Dividir em grupos de 500 e tirando a media, {scaler.title()})")
    plt.show()

mse_df = pd.DataFrame.from_dict(dataframe_mse, orient='index', columns=['without-scaler', 'min-max', 'z-score'])
mae_df = pd.DataFrame.from_dict(dataframe_mae, orient='index', columns=['without-scaler', 'min-max', 'z-score'])
r2_df = pd.DataFrame.from_dict(dataframe_r2, orient='index', columns=['without-scaler', 'min-max', 'z-score'])

display('MSE', mse_df)
display('MAE', mae_df)
display('R2', r2_df)

# %%

# Árvore de Decisão

dataframe_mse = {}
dataframe_mae = {}

for scaler in scalers:
    if(scalers[scaler] == None):
        pipe = Pipeline([('regressor', DecisionTreeRegressor())])
    else:
        pipe = Pipeline([(scaler, scalers[scaler]), ('regressor', DecisionTreeRegressor())])

    pipe.fit(X_train, y_train)
    y_predict = pipe.predict(X_test)

    mse = mean_squared_error(y_test, y_predict)
    mae = mean_absolute_error(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)

    key = f'metric'

    if key in dataframe_mse:
        dataframe_mse[key].append(mse)
        dataframe_mae[key].append(mae)
    else:
        dataframe_mse[key] = [mse]
        dataframe_mae[key] = [mae]

    y_test_media = []
    y_predict_media = []
    tamanhoGrupo = 500
    for i in range(0, len(y_test), tamanhoGrupo):
        y_predict_media.append(np.mean(y_predict[i:i + tamanhoGrupo]))
        y_test_media.append(np.mean(y_test[i:i + tamanhoGrupo]))

    plt.scatter(y_predict_media, y_test_media)
    plt.xlabel("Predict")
    plt.ylabel("Test")
    plt.title(
        f"Gráfico de Dispersão(Dividir em grupos de 500 e tirando a media, {scaler.title()})")
    plt.show()

    plt.scatter(y_predict, y_test)
    plt.xlabel("Predict")
    plt.ylabel("Test")
    plt.title(
        f"Gráfico de Dispersão(SEM Dividir em grupos de 500 e tirando a media, {scaler.title()})")
    plt.show()

mse_df = pd.DataFrame.from_dict(dataframe_mse, orient='index', columns=['without-scaler', 'min-max', 'z-score'])
mae_df = pd.DataFrame.from_dict(dataframe_mae, orient='index', columns=['without-scaler', 'min-max', 'z-score'])

display('MSE', mse_df)
display('MAE', mae_df)

# %%

# Random Forest

dataframe_mse = {}
dataframe_mae = {}

for scaler in scalers:
    if(scalers[scaler] == None):
        pipe = Pipeline([('regressor', DecisionTreeRegressor())])
    else:
        pipe = Pipeline([(scaler, scalers[scaler]), ('regressor', DecisionTreeRegressor())])

    pipe.fit(X_train, y_train)
    y_predict = pipe.predict(X_test)

    mse = mean_squared_error(y_test, y_predict)
    mae = mean_absolute_error(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)

    key = f'metric'

    if key in dataframe_mse:
        dataframe_mse[key].append(mse)
        dataframe_mae[key].append(mae)
    else:
        dataframe_mse[key] = [mse]
        dataframe_mae[key] = [mae]

    y_test_media = []
    y_predict_media = []
    tamanhoGrupo = 500
    for i in range(0, len(y_test), tamanhoGrupo):
        y_predict_media.append(np.mean(y_predict[i:i + tamanhoGrupo]))
        y_test_media.append(np.mean(y_test[i:i + tamanhoGrupo]))

    plt.scatter(y_predict_media, y_test_media)
    plt.xlabel("Predict")
    plt.ylabel("Test")
    plt.title(
        f"Gráfico de Dispersão(Dividir em grupos de 500 e tirando a media, {scaler.title()})")
    plt.show()

    plt.scatter(y_predict, y_test)
    plt.xlabel("Predict")
    plt.ylabel("Test")
    plt.title(
        f"Gráfico de Dispersão(SEM Dividir em grupos de 500 e tirando a media, {scaler.title()})")
    plt.show()

mse_df = pd.DataFrame.from_dict(dataframe_mse, orient='index', columns=['without-scaler', 'min-max', 'z-score'])
mae_df = pd.DataFrame.from_dict(dataframe_mae, orient='index', columns=['without-scaler', 'min-max', 'z-score'])

display('MSE', mse_df)
display('MAE', mae_df)

# %%

# Rede Neural MLP

dataframe_mse = {}
dataframe_mae = {}

for scaler in scalers:
    if(scalers[scaler] == None):
        pipe = Pipeline([('regressor', MLPRegressor())])
    else:
        pipe = Pipeline([(scaler, scalers[scaler]), ('regressor', MLPRegressor())])

    pipe.fit(X_train, y_train)
    y_predict = pipe.predict(X_test)

    mse = mean_squared_error(y_test, y_predict)
    mae = mean_absolute_error(y_test, y_predict)

    key = f'metric'

    if key in dataframe_mse:
        dataframe_mse[key].append(mse)
        dataframe_mae[key].append(mae)
    else:
        dataframe_mse[key] = [mse]
        dataframe_mae[key] = [mae]

mse_df = pd.DataFrame.from_dict(dataframe_mse, orient='index', columns=['without-scaler', 'min-max', 'z-score'])
mae_df = pd.DataFrame.from_dict(dataframe_mae, orient='index', columns=['without-scaler', 'min-max', 'z-score'])

display('MSE', mse_df)
display('MAE', mae_df)

# %%

# Comparação entre os Métodos

dataframe_mse = {}
dataframe_mae = {}

for regressor in regressors:
    for scaler in scalers:
        if(scalers[scaler] == None):
            pipe = Pipeline([('regressor', regressors[regressor])])
        else:
            pipe = Pipeline([(scaler, scalers[scaler]), ('regressor', regressors[regressor])])

        pipe.fit(X_train, y_train)
        y_predict = pipe.predict(X_test)

        mse = mean_squared_error(y_test, y_predict)
        mae = mean_absolute_error(y_test, y_predict)

        if regressor in dataframe_mse:
            dataframe_mse[regressor].append(mse)
            dataframe_mae[regressor].append(mae)
        else:
            dataframe_mse[regressor] = [mse]
            dataframe_mae[regressor] = [mae]

mse_df = pd.DataFrame.from_dict(dataframe_mse, orient='index', columns=['without-scaler', 'min-max', 'z-score'])
mae_df = pd.DataFrame.from_dict(dataframe_mae, orient='index', columns=['without-scaler', 'min-max', 'z-score'])

display('MSE', mse_df)
display('MAE', mae_df)