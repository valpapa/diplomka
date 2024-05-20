import pandas as pd

df = pd.read_excel('Data_230406_small.xlsx', sheet_name='Data 1')


%%time

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler, PowerTransformer, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import warnings

from sklearn.pipeline import make_pipeline






target = 'YIELD_STRENGTH_TARGET'

features = ['CHEMISTRY_INDX0', 'CHEMISTRY_INDX1', 'CHEMISTRY_INDX2', 'CHEMISTRY_INDX3', 'CHEMISTRY_INDX4', 
            'CHEMISTRY_INDX5', 'CHEMISTRY_INDX6', 'CHEMISTRY_INDX7', 'CHEMISTRY_INDX8', 'CHEMISTRY_INDX9'] 

features_with_strain = features + ['NN_STRAIN']


df = df.drop_duplicates(subset=features_with_strain)
df = df.dropna(subset=features_with_strain)

dataframe=df



def log_transform(x):
    return np.log1p(x+1)

def inverse_transform(x):
    return (1/(x+0.0001))

def normalize_data(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

# Vytvoření instance FunctionTransformer
log_transformer = FunctionTransformer(log_transform)
inverse_transformer = FunctionTransformer(inverse_transform)
normalize_transformer = FunctionTransformer(normalize_data)

predzpracovani = {
    
    'PowerTransformer': PowerTransformer(method='yeo-johnson', standardize=True),
    'LogTransformer': log_transformer,
    'MinMaxScaler': MinMaxScaler(),
    'InverseTransformer': inverse_transformer,
    'Normalize': normalize_transformer
}
# Modely
modely = {
    'KNeighborsRegressor':KNeighborsRegressor(n_neighbors=60,weights='distance',metric='manhattan'),#(metric='minkowski',n_neighbors=105,weights='distance')# KNeighborsRegressor(),
    'GradientBoostingRegressor': GradientBoostingRegressor(),
    'XGBRegressor': XGBRegressor(),
    'RandomForestRegressor': RandomForestRegressor(),
    'DecisionTreeRegressor': DecisionTreeRegressor(),
    'ExtraTreesRegressor':ExtraTreesRegressor(),# ExtraTreesRegressor(max_depth=50,max_features=0.2987673199120496, min_samples_leaf= 34, min_samples_split=6, n_estimators= 19)#
    'MLPRegressor': MLPRegressor(hidden_layer_sizes=(317,), activation='relu', solver='adam', alpha=0.0001,
                                 batch_size='auto', learning_rate='adaptive', learning_rate_init=0.1, #0.01   constant
                                 max_iter=10, random_state=42, tol=0.0001, verbose=False), #False 20
    'LinearRegression': LinearRegression()
}


for pred_name, pred in predzpracovani.items():
    print('Nové předzpracování')
    for model_name, model in modely.items():

        mse_list = []

        length_list=[]

        mean_percentage_error_list = []

        mse_list_ap = []

        mean_percentage_error_list_ap = []


        unique_rows = np.unique(df[features], axis=0)
        unique_alloys = df['ALLOY'].unique()

        i=0

        for test_features in unique_rows:#unique_rows:unique_alloys



            mask = (dataframe[features] == test_features).all(axis=1)
    #mask = (dataframe['ALLOY'] == test_features)

            test_df = dataframe[mask]
            train_df = dataframe[~mask]

            X_train,y_train = train_df[features_with_strain],train_df[target]
            X_test,y_test = test_df[features_with_strain],test_df[target]


            indices_to_move = X_test.sample(n=0, random_state=42).index  #množství dat přesunutých z testovací množiny do trénovací 

    # Přidejte vybrané řádky do trénovací sady
            X_train = pd.concat([X_train, X_test.loc[indices_to_move]])
            y_train = pd.concat([y_train, y_test.loc[indices_to_move]])

# Odstraňte vybrané řádky z testovací sady
            X_test = X_test.drop(indices_to_move)
            y_test = y_test.drop(indices_to_move)
            
            ''''''
            pipeline = Pipeline([
            ('scaler', pred),
            ('model', model)
        ])
            
            pipeline.fit(X_train,y_train)
            y_pred = pipeline.predict(X_test)


            mse = mean_squared_error(y_test, y_pred)


            mse_list.append(mse)
            length_list.append(len(y_test))

            

    # Výpočet procentuální odchylky
            percentage_error = np.abs((y_test - y_pred) / y_test) * 100
            mean_percentage_error = np.mean(percentage_error)
            mean_percentage_error_list.append(mean_percentage_error)
            #print(mean_percentage_error)
            


            X_test_krivka=X_test.copy()

                #t = np.linspace(min(X_test['NN_STRAIN'])-0.5, max(X_test['NN_STRAIN']+1.0), len(X_test['NN_STRAIN']))
            t = np.linspace(0, 2.5, len(X_test['NN_STRAIN']))
                
            X_test_krivka.loc[:, 'NN_STRAIN'] = t

            y_krivka = pipeline.predict(X_test_krivka)
            X =t.reshape(-1, 1) #y_pred.reshape(-1, 1)  # Převod y_pred na 2D array pro fit_transform
            y = y_krivka#.reshape(-1, 1)#y_test  # Skutečné hodnoty odpovídající X_test
            degree = 3

        # Vytvoření a trénování polynomiálního regresního modelu
            poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
            poly_model.fit(X, y)

            X_fit = np.linspace(min(X), max(X), 1000).reshape(-1, 1)
            y_fit = poly_model.predict(X_fit)

            y_pred_ap = poly_model.predict((X_test['NN_STRAIN'].values.reshape(-1, 1)))

            mse_ap = mean_squared_error(y_test, y_pred_ap)


            mse_list_ap.append(mse_ap)

            percentage_error_ap = np.abs((y_test - y_pred_ap) / y_test) * 100
            mean_percentage_error_ap = np.mean(percentage_error_ap)
            mean_percentage_error_list_ap.append(mean_percentage_error_ap)

            #print(f"Mape: {mean_percentage_error:.2f} Mape ap: {mean_percentage_error_ap:.2f}")


            
            i+=1

            #if mean_percentage_error>0 or mean_percentage_error<0 and i==1:
            if  i==1:
                X_test_krivka=X_test.copy()

                #t = np.linspace(min(X_test['NN_STRAIN'])-0.5, max(X_test['NN_STRAIN']+1.0), len(X_test['NN_STRAIN']))
                t = np.linspace(0, 2.5, len(X_test['NN_STRAIN']))
                
                X_test_krivka.loc[:, 'NN_STRAIN'] = t

                y_krivka = pipeline.predict(X_test_krivka)
                plt.figure(figsize=(5, 3.8))
        #plt.scatter(X_test['NN_STRAIN'], y_pred, alpha=0.9,label='Predikovaná data')
                plt.scatter(X_test['NN_STRAIN'], y_test, alpha=0.9,label='Skutečná data',s=10)
        #plt.scatter(t, y_krivka, alpha=1,label='Predikce',s=5,marker='x')
                plt.plot(X_test_krivka['NN_STRAIN'], y_krivka, alpha=1,label='Predikce',color='orange')


                X =t.reshape(-1, 1) #y_pred.reshape(-1, 1)  # Převod y_pred na 2D array pro fit_transform
                y = y_krivka#.reshape(-1, 1)#y_test  # Skutečné hodnoty odpovídající X_test
                degree = 3

        # Vytvoření a trénování polynomiálního regresního modelu
                poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
                poly_model.fit(X, y)

                X_fit = np.linspace(min(X), max(X), 1000).reshape(-1, 1)
                y_fit = poly_model.predict(X_fit)

                linear_reg = poly_model.named_steps['linearregression']

    # Získání koeficientů (váhy) a konstantního členu (intercept)
                coefficients = linear_reg.coef_
                intercept = linear_reg.intercept_


# Pro sestavení předpisu polynomu potřebujeme také znát stupně vlastností
# Tyto informace můžeme získat z prvního kroku pipeline, což je PolynomialFeatures
                poly_features = poly_model.named_steps['polynomialfeatures']
                feature_names = poly_features.get_feature_names_out(input_features=['x'])

# Sestavení a vytisknutí předpisu polynomu
    
                polynomial_formula = f"{intercept:.2f}"
                for coef, name in zip(coefficients, feature_names):
                    if coef > 0:
                        polynomial_formula += f" + {coef:.2f}*{name}"
                    elif coef < 0:
                        polynomial_formula += f" - {abs(coef):.2f}*{name}"


                #print(mean_percentage_error_ap)
                
                plt.plot(X_fit, y_fit, label=f'Polynomiální fit: stupeň {degree}', color='red')
                plt.xlabel('Strain')
                plt.ylabel('Target [GPa]')
                plt.title('Závislost strain a target')
                plt.legend()
                #plt.savefig(f'obr\graf_loo_poly_{model_name}.png', bbox_inches='tight')
        #plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Diagonální čára
                plt.show()
                #break
                

        
        print(f"{pred_name} + {model_name}:")
        print(f"Střední kvadratická chyba (MSE): {np.mean(mse_list):.2f}")
        print(f"Průměrná procentuální odchylka: {np.mean(mean_percentage_error_list):.2f}%")
        print(f"Nejvyšší procentuální odchylka : {np.max(mean_percentage_error_list):.2f}%")
        print()
        print(f"Střední kvadratická chyba (MSE) aproximace: {np.mean(mse_list_ap):.2f}")
        print(f"Průměrná procentuální odchylka aproximace: {np.mean(mean_percentage_error_list_ap):.2f}%")
        print(f"Nejvyšší procentuální odchylka aproximace: {np.max(mean_percentage_error_list_ap):.2f}%")

 
