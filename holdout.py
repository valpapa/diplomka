import pandas as pd

df = pd.read_excel('Data_230406_small.xlsx', sheet_name='Data 1')


%%time
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor,    ExtraTreesRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import StandardScaler

import xgboost 
from xgboost import XGBRegressor

import warnings

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures

from sklearn.preprocessing import PowerTransformer

def log_transform(x):
    return np.log1p(x)  # log1p je bezpečnější pro práci s nulami a kladnými hodnotami

# Vytvoření instance FunctionTransformer
log_transformer = FunctionTransformer(log_transform)


warnings.filterwarnings("ignore")

target = 'YIELD_STRENGTH_TARGET'

features = ['CHEMISTRY_INDX0', 'CHEMISTRY_INDX1', 'CHEMISTRY_INDX2', 'CHEMISTRY_INDX3', 'CHEMISTRY_INDX4', 
            'CHEMISTRY_INDX5', 'CHEMISTRY_INDX6', 'CHEMISTRY_INDX7', 'CHEMISTRY_INDX8', 'CHEMISTRY_INDX9'] 

features_with_strain = features + ['NN_STRAIN']

# Předpokládáme, že unique_rows a ostatní potřebné proměnné jsou již definovány
X_train_all = pd.DataFrame(columns=features_with_strain)
y_train_all = pd.Series(name=target)

X_test_all = pd.DataFrame(columns=features_with_strain)
y_test_all = pd.Series(name=target)

X_val_all = pd.DataFrame(columns=features_with_strain)
y_val_all = pd.Series(name=target)

#y_train1 = pd.DataFrame(columns=target)

mean_percentage_error_list = []
mse_list = []

dataframe=df

unique_rows = np.unique(df[features], axis=0)

for test_features in unique_rows:  # Iterace přes každý unikátní řádek
    # Vytvoření masky pro selekci dat na základě unikátního řádku
    mask = (dataframe[features] == test_features).all(axis=1)

    # Data pro aktuální unikátní řádek
    data = dataframe[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        data[features_with_strain], data[target], test_size=0.2, random_state=42
    )

    X_train_all = pd.concat([X_train_all,X_train])
    y_train_all = pd.concat([y_train_all, y_train])

    X_test_all = pd.concat([X_test_all,X_test])
    y_test_all = pd.concat([y_test_all, y_test])


modely = {
    'KNeighborsRegressor': KNeighborsRegressor(),
    'GradientBoostingRegressor': GradientBoostingRegressor(),
    'XGBRegressor': XGBRegressor(),
    'RandomForestRegressor': RandomForestRegressor(),
    'DecisionTreeRegressor': DecisionTreeRegressor(),
    'ExtraTreesRegressor': ExtraTreesRegressor(),
    'MLP':MLPRegressor(hidden_layer_sizes=(317,), activation='relu', solver='adam', alpha=0.0001,
                   batch_size='auto', learning_rate='constant', learning_rate_init=0.01,
                   max_iter=1000, random_state=42, tol=0.0001, verbose=False),
    'LinearRegresion': LinearRegression()
}

predzpracovani=[PowerTransformer(method='yeo-johnson', standardize=True),log_transformer(),MinMaxScaler(),inverzni_trasf(),normlizace()]

for pred in predzpracovani.items():
    print(pred)
    for name_model, model in modely.items():
        pipeline = Pipeline([
            ('scaler', PowerTransformer(method='yeo-johnson', standardize=True)), # log_transformer()), # MinMaxScaler()),
            ('model', model_model)
    ])

        pipeline.fit(X_train_all, y_train_all)
        y_pred = pipeline.predict(X_test_all)
        mse = mean_squared_error(y_test_all, y_pred)
        mean_percentage_error = np.mean(np.abs((y_test_all - y_pred) / y_test_all) * 100)

        print(name_model)
        print(f"Celkové MSE : {mse:.2f}")
        print(f"Celkové MAPE : {mean_percentage_error:.2f}%")
        print()
