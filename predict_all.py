# %%

import pandas as pd
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

ratings = pd.read_csv('dataset_small/ratings.csv')

display(ratings)

# %%

# Sem a coluna de Géneros

X = ratings.drop(columns=['rating', 'timestamp'])
y = ratings['rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

random_row = X_test.sample(n=1, random_state=42)
display(ratings.loc[(ratings['movieId'] == random_row.iloc[0]['movieId']) & (ratings['userId'] == random_row.iloc[0]['userId'])])

dataframe_predict = {}

for regressor in regressors:
    for scaler in scalers:
        if(scalers[scaler] == None):
            pipe = Pipeline([('regressor', regressors[regressor])])
        else:
            pipe = Pipeline([(scaler, scalers[scaler]), ('regressor', regressors[regressor])])

        pipe.fit(X_train, y_train)
        y_predict = pipe.predict(random_row)

        if regressor in dataframe_predict:
            dataframe_predict[regressor].append(y_predict[0])
        else:
            dataframe_predict[regressor] = [y_predict[0]]

predict_df = pd.DataFrame.from_dict(dataframe_predict, orient='index', columns=['without-scaler', 'min-max', 'z-score'])

display(predict_df)

# %%

# Com a coluna de géneros

movies = pd.read_csv('dataset_small/movies.csv')

merged = ratings.merge(movies[['movieId', 'genres']], on='movieId', how='left')

genres_encoded = merged['genres'].str.get_dummies(sep='|')
data = pd.concat([merged, genres_encoded], axis=1)
data.drop('genres', axis=1, inplace=True)

display(data)

# %%

X = data.drop(columns=['rating', 'timestamp'])
y = data['rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

random_row = X_test.sample(n=1, random_state=42)
display(merged.loc[(data['movieId'] == random_row.iloc[0]['movieId']) & (data['userId'] == random_row.iloc[0]['userId'])])

dataframe_predict = {}

for regressor in regressors:
    for scaler in scalers:
        if(scalers[scaler] == None):
            pipe = Pipeline([('regressor', regressors[regressor])])
        else:
            pipe = Pipeline([(scaler, scalers[scaler]), ('regressor', regressors[regressor])])

        pipe.fit(X_train, y_train)
        y_predict = pipe.predict(random_row)

        if regressor in dataframe_predict:
            dataframe_predict[regressor].append(y_predict[0])
        else:
            dataframe_predict[regressor] = [y_predict[0]]

predict_df = pd.DataFrame.from_dict(dataframe_predict, orient='index', columns=['without-scaler', 'min-max', 'z-score'])

display(predict_df)