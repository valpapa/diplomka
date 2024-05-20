import pandas as pd


df = pd.read_excel('Data_230406_small.xlsx', sheet_name='Data 1')


import matplotlib.pyplot as plt
import numpy as np


data=df2['NN_STRAIN']
plt.hist(data, bins=100, edgecolor='black')

# Popisky os a nadpis
plt.xlabel('Hodnoty')
plt.ylabel('Četnost')
plt.title('Histogram hodnot strain')
#plt.savefig(f'obr\hist_strian_big.png')

# Zobrazení grafu
plt.show()


data=df2['YIELD_STRENGTH_TARGET']
plt.hist(data, bins=100, edgecolor='black')

# Popisky os a nadpis
plt.xlabel('Hodnoty')
plt.ylabel('Četnost')
plt.title('Histogram hodnot target')
#plt.savefig(f'obr\hist_target_big.png')

# Zobrazení grafu
plt.show()




#odstranění outliers, duplicitních hodnot a nekompletních dat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import seaborn as sns
from sklearn.neural_network import MLPRegressor



features = ['CHEMISTRY_INDX0', 'CHEMISTRY_INDX1', 'CHEMISTRY_INDX2', 'CHEMISTRY_INDX3', 'CHEMISTRY_INDX4', 'CHEMISTRY_INDX5', 'CHEMISTRY_INDX6', 'CHEMISTRY_INDX7', 'CHEMISTRY_INDX8', 'CHEMISTRY_INDX9']
target = 'YIELD_STRENGTH_TARGET'
features_with_strain = features + ['NN_STRAIN']
all_features = features_with_strain + [target]
unique_rows = np.unique(df[features], axis=0)

y=df[target]
dataframe=df[all_features] #features_with_strain

unique_rows = np.unique(dataframe[features], axis=0)

p=0

for test_features in unique_rows:#unique_rows:unique_alloys
    mask = (dataframe[features] == test_features).all(axis=1)

    test_df = dataframe[mask]
    

    nejmensi_hodnota=min(test_df['NN_STRAIN'])
    nejvetsi_hodnota=max(test_df['NN_STRAIN'])
    body=[min(test_df['NN_STRAIN'])]
    shluk_list= []

    hranice = 0.1

    for index, row in test_df.iterrows():
        far_enough = True  # Předpokládáme, že je bod dostatečně daleko, dokud se neprokáže opak
        for bod in body:
            if row['NN_STRAIN'] > bod - hranice and row['NN_STRAIN'] < bod + hranice:
                far_enough = False  # Nalezli jsme bod, který je příliš blízko
                break  # Není třeba dále testovat, protože už víme, že není dostatečně daleko

        if far_enough:
            body.append(row['NN_STRAIN'])

    for index, row in test_df.iterrows():
        for bod in body:
            if row['NN_STRAIN'] > bod - hranice and row['NN_STRAIN'] < bod + hranice:

                shluk_list.append(body.index(bod))
                break

    dataframe.loc[test_df.index, 'shluk'] = shluk_list

    int_list = test_df[target].astype(int).tolist()



# Použijeme defaultdict k vytvoření seznamu pro každou novou kategorii
    categories = defaultdict(list)


    for i, label in enumerate(shluk_list):
        categories[label].append(int_list[i])

    #data=[nula,jedna]
    data = list(categories.values())
    

 # List obsahující data všech skupin
    body = [round(x, 1) for x in body]


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    slitina = dataframe[mask]
    slitina_target = y[mask]
    ax1.scatter(slitina['NN_STRAIN'], slitina_target.values, label='Sine',alpha=0.9,s=5)#alpha=0.2
    #ax1.plot(t,y_pred,color='red',label='Příklad predikce')
    #ax1.hexbin(slitina['NN_STRAIN'], slitina_target.values, gridsize=30, cmap='Blues')
    #sns.kdeplot(x=slitina['NN_STRAIN'], y=slitina_target.values, cmap='Blues', fill=True, thresh=0, levels=100)
    ax1.set_title('Skutečná data')
    ax1.set_xlabel('Strain')
    ax1.set_ylabel('Target [GPa]')
    #ax1.legend()

    ax2.boxplot(data,labels=body,positions=body, patch_artist=True,vert=True)
    ax2.set_title('Boxplot')
    ax2.set_xlabel('Strain')
    ax2.set_ylabel('Target [GPa]')
    #ax2.legend()
    plt.tight_layout()
   # plt.savefig(f'obr\strain_boxploty{p+1}.png')
    plt.show()

    p+=1
    if p==20:
        break

    
    
    




        







