# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 12:23:26 y_

@author: jst
"""


import pandas as pd

import math
# import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
# from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
# from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import json
import shap

def get_new_data(fractile):
    path_str = 'R:\\Ráðgjöf\\Bláa hagkerfið\\Hafró\\distribution_output\\'
    path_str_do = 'R:\\Ráðgjöf\\Maris Optimum/distribution_output\\'
    X_df = pd.read_csv(path_str_do+'distribution'+fractile+'.csv',
                   sep=",")
    
    catch_df = pd.read_csv(path_str+'golden_redfish_catch.csv',
                           sep=";")
    print(catch_df)
    catch_df.at[37, 'catch'] = 26
    
    print(catch_df)
    
    X_cal_df = pd.read_csv(path_str_do+'distribution_commercial.csv',
                           sep=",")
    X_cal_df.drop(1605, axis = 0, inplace = True)
    
    X_cal_df = X_cal_df.pivot(index='ar',
                              columns='lengd',
                              values='per_length')

    X_cal_df = X_cal_df.fillna(0)
    X_cal_df.columns = 1000 + X_cal_df.columns
    #X_cal_df.index 
    X_cal_df.columns = X_cal_df.columns.astype(int).astype(str)
    
    XX_df = X_df.pivot(index='ar',
                       columns='lengd',
                       values='per_length')
    
   # XX_df.drop(4.5, axis=1, inplace=True)
   # XX_df.drop(6.4, axis=1, inplace=True)
    XX_df.drop(11.9, axis=1, inplace=True)
    XX_df.drop(12.5, axis=1, inplace=True)
    XX_df.drop(12.6, axis=1, inplace=True)
    XX_df.drop(13.1, axis=1, inplace=True)
    XX_df.drop(13.4, axis=1, inplace=True)
    XX_df.drop(13.6, axis=1, inplace=True)
    XX_df.drop(13.7, axis=1, inplace=True)
    XX_df.drop(13.9, axis=1, inplace=True)
    XX_df.drop(14.4, axis=1, inplace=True)
    XX_df.drop(14.5, axis=1, inplace=True)
    XX_df.drop(14.7, axis=1, inplace=True)
    XX_df.drop(14.8, axis=1, inplace=True)
    XX_df.drop(14.9, axis=1, inplace=True)
    
    XX_df.columns = XX_df.columns.astype(int).astype(str)

    YX = pd.read_csv(path_str+"RED_numbers_at_age.csv", sep=";")
    YY = YX.iloc[15:53, 28]
    
    catch_df = catch_converter(X_cal_df, catch_df)
    print(catch_df)

    
    XX_df['catch'] = catch_df.number.values * -1000000
    XX_df = XX_df.join(X_cal_df.iloc[:, :])
    XX_df = XX_df.fillna(0)
    XX_df.index =XX_df.index.astype(str)  
    return (XX_df, YY)

def catch_converter(X_catch_per_df, catch_df ):
    '''

     '''        
    for ind in range(1985,2022):
        average_weight = 0
        for col in range(1010,1060):
            average_weight += ( X_catch_per_df.loc[ind, str(col)])* (0.0015*(col-1000)**2 - 0.055*(col-1000) + 0.65)
        catch_df.at[ind+1-1985, 'number'] = catch_df.loc[ind+1-1985,'catch']/average_weight
    return catch_df
    

def get_data(fractile):
    path_str = 'R:\\Ráðgjöf\\Bláa hagkerfið\\Hafró\\distribution_output\\'
    path_str_mo = 'R:\\Ráðgjöf\\Maris Optimum/Golden_redfish_model\\'
    X_df = pd.read_csv(path_str+'distribution'+fractile+'.csv',
                   sep=",")
    catch_df = pd.read_csv(path_str+'golden_redfish_catch.csv',
                           sep=";")
    catch_df.at[2022] = 26
    
    X_cal_df = pd.read_csv(path_str_mo+'distribution_commercial.csv',
                           sep=',')
    X_cal_df = X_cal_df.pivot(index='ar',
                              columns='lengd',
                              values='percent_per_year')
    X_cal_df = X_cal_df.fillna(0)
    X_cal_df.columns = 1000 + X_cal_df.columns
    X_cal_df.index += 1
    X_cal_df.columns = X_cal_df.columns.astype(int).astype(str)
    
    XX_df = X_df.pivot(index='ar',
                       columns='lengd',
                       values='sum_fjoldi')
    
    XX_df.drop(6.4, axis=1, inplace=True)
    XX_df.drop(11.9, axis=1, inplace=True)
    XX_df.drop(12.5, axis=1, inplace=True)
    XX_df.drop(12.6, axis=1, inplace=True)
    XX_df.drop(13.1, axis=1, inplace=True)
    XX_df.drop(13.4, axis=1, inplace=True)
    XX_df.drop(13.6, axis=1, inplace=True)
    XX_df.drop(13.7, axis=1, inplace=True)
    XX_df.drop(13.9, axis=1, inplace=True)
    XX_df.drop(14.4, axis=1, inplace=True)
    XX_df.drop(14.5, axis=1, inplace=True)
    XX_df.drop(14.7, axis=1, inplace=True)
    XX_df.drop(14.8, axis=1, inplace=True)
    XX_df.drop(14.9, axis=1, inplace=True)
    
    XX_df.columns = XX_df.columns.astype(int).astype(str)
    
    
    XX_2022_dict={"lengd": [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60, 'catch'],
                  "y": [0,0,1.73,2.796,3.405,2.969,3.761,2.299,2.299,1.805,1.191,1.724,1.724,1.724,2.24,2.299,2.299,1.737,2.21,2.299,2.299,2.902,3.448,3.448,3.448,2.874,2.939,3.448,3.448,7.011,15.238,40.273,68.759,74.023,78.7,51.832,37.509,28.046,20.581,12.794,8.02,5.445,3.815,2.938,2.38,0,0,0,0,0,0,0,0,0,0,0,0]}
    sum_2022_y = sum(XX_2022_dict['y'])
    for i in range(5, 60):
        XX_df.at[2022, str(i)] = XX_2022_dict['y'][i-5]/sum_2022_y*XX_df.iloc[24, :56].sum()*.85
    
    
    YX = pd.read_csv(path_str+"RED_numbers_at_age.csv", sep=";")
    YY = YX.iloc[15:53, 28]
    
    
    XX_df['catch'] = catch_df.catch.values*-1000
    XX_df = XX_df.join(X_cal_df.iloc[:, :])
    XX_df = XX_df.fillna(0)
    


    '''
    XH_df = pd.read_csv(path_str+'distributionH'+fractile+'.csv',
                        sep=",")
    
    XXH_df = XH_df.pivot(index='ar',
                         columns='lengd',
                         values='sum_fjoldi')
    
    
    XXH_df.columns = XX_df.columns.astype(int).astype(str)
    XXH_df = XXH_df.drop(2011)
    XXH_df = XXH_df.iloc[1:25, :]
    

    
    YXH = pd.read_csv(path_str+"RED_smh.csv", sep=",")
    YXH['sum'] = YXH.iloc[:, 3:29].sum(axis=1)*1e6
    
    

    

    
    XXH_df['catch'] = catch_df.catch.iloc[12:36].values*-1000
    XXH_df.at['2021', 'catch'] = -34000
    
    XXH_2021_dict = {"x": [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54],
                     "y": [0.125,0.125,0.201,0.708,0.457,0.374,0.374,0.453,0.811,0.201,0.943,0.667,0.499,0.499,1.123,1.123,1.74,2.121,3.679,9.917,23.615,39.645,46.734,50.748,43.742,31.137,30.46,14.397,10.982,6.198,3.41,1.929,0.883,0.748,0.748,0.748,0.844,1.097,0.815]}
    
    sum_2021H_y = sum(XXH_2021_dict['y'])
    
    for i in range(16, 60):
        XXH_df.at['2021', str(i)] = XX_2022_dict['y'][i-16] / sum_2022_y * XXH_df.iloc[23, :56].sum()
    XXH_df = XXH_df.join(X_cal_df.iloc[:, :])
    XXH_df = XXH_df.fillna(0)
    
    # begin autumnal data
    
    
    XX_df=XXH_df.iloc[1:, :]
    YY=(YY[14:])
    '''
    # end autumnal data
    return (XX_df, YY)

def fitting_plot(y_test, y_pred_test, X_test, xgb_regressor):
    x_ax= range(len(y_test))
    plt.scatter(x_ax,
                y_test,
                s=5,
                color="blue",
                label="original")
    plt.plot(x_ax,
             y_pred_test,
             lw=0.8,
             color="red",
             label="predicted")
    plt.xticks(x_ax,
               X_test.index,
               rotation = 45)
    plt.grid(True)
    plt.legend()
    plt.show()
    
def error_plot(regressor):
    results = regressor.evals_result()
    epochs = len(results['validation_0']['mae'])
    x_axis = range(0, epochs)
    fig, ax = plt.subplots()
    plt.grid(True, which='major')
    ax.plot(x_axis, results['validation_0']['mae'],
            label='Train')
    ax.plot(x_axis, results['validation_1']['mae'],
            label='Test')
    ax.legend()
    plt.ylabel('Error')
    plt.title('Error')
    plt.show()

def shap_calculations_xgb(regressor,XX_df):
    
    
    explainer = shap.TreeExplainer(regressor)
    
    shap_values = explainer.shap_values(XX_df.iloc[16:,:])
    
    shap.summary_plot(shap_values,
                      XX_df.iloc[16:,:],
                      plot_type="layered_violin",
                      max_display=10)
    
    shap_values = explainer(XX_df.iloc[16:,:])
    shap.waterfall_plot(shap_values[21])

def shap_calculations_rf(regressor,XX_df):
        
    X = XX_df.iloc[16:,:]
    explainer = shap.TreeExplainer(regressor)
    
    shap_values = explainer.shap_values(X)
    
    shap.summary_plot(shap_values,
                      X,
                      plot_type="layered_violin",
                      max_display=10)
    
    class helper_object():
        def __init__(self, i):
            self.base_values = shap_values.base_values[i][0]
            self.data = shap_values.data[i]
            self.feature_names = X.columns.to_list()
            self.values = shap_values.values[i]
    shap_values = explainer(X)
    
    shap.waterfall_plot(helper_object(21))
    
    
    
    
    '''
    X50 = shap.utils.sample(XX_df, 50)
    explainer = shap.TreeExplainer(regressor.predict, X50)
    shap_values = explainer(X50)
    
    
    sample_ind = 33
    shap.partial_dependence_plot(
        "1035", regressor.predict, X50, model_expected_value=True,
        feature_expected_value=True, 
        ice=False,
        shap_values=shap_values[sample_ind:sample_ind+1,:])
    

    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(X50)
    shap.dependence_plot(
        "catch",
        shap_values,
        X50)
'''

def regression_over_possible_values_XGB(X,y, interval_int):
    parameters = {
        'nthread': [0],
        'objective': ['reg:squarederror'],
        'eval_metric': ["error"],
        'learning_rate': [.4, .5, .6, .7, .8],
        'max_depth': [3, 4, 5],
        'min_child_weight': [3, 4, 5],
        'subsample': [0.6, 0.7],
        'colsample_bytree': [.5, .6, .7],
        'n_estimators': [35]
    }
    test_size = .30
    seed = 4
    result_dict = {'fjoldi': [], 'mae': [], 'rmse': [], 'r2': [], 'evs': []}

    xgb1 = xgb.XGBRegressor(seed)
    
    for add_int in range(0, 200000000, interval_int):
    
        X_train, X_test, y_train, y_test = train_test_split(X.iloc[14:, :],
                                                            y.iloc[14:],
                                                            test_size=test_size,
                                                            random_state=seed)
    
    
        dtrain = dtrain = xgb.DMatrix(data=X_train.values,
                     feature_names=X_train.columns,
                     label=y_train.values)
    
        xgb_regressor = GridSearchCV(xgb1,
                                     parameters,
                                     cv=2,
                                     n_jobs=5,
                                     verbose=0)
    
        xgb_regressor.fit(X, y)
    
        print(json.dumps(xgb_regressor.best_params_, sort_keys=False, indent=4))
    
        params = xgb_regressor.best_params_
        eval_set = [(X_train, y_train), (X_test, y_test)]
        xgb_regressor = xgb.XGBRegressor(**params)
    
        xgb_regressor.fit(X_train,
                          y_train,
                          eval_metric=["mae"],
                          eval_set=eval_set,
                          verbose=False)
    
        y_pred_test = xgb_regressor.predict(X_test)

    
        result_dict['fjoldi'].append(y[52])
        result_dict['mae'].append(mean_absolute_error(y_test,
                                                      y_pred_test))
        result_dict['rmse'].append(math.sqrt(mean_squared_error(y_test,
                                                                y_pred_test)))
        result_dict['r2'].append(r2_score(y_test,
                                          y_pred_test))
        result_dict['evs'].append(explained_variance_score(y_test,
                                                           y_pred_test))
        y[50] += interval_int
        y[51] += interval_int
        y[52] += interval_int
    y[50] = 515000000
    y[51] = 475000000
    y[52] = 435000000
    regressor = GridSearchCV(xgb1,
                                     parameters,
                                     cv=2,
                                     n_jobs=5,
                                     verbose=0)
    
    regressor.fit(X, y)
    params = regressor.best_params_
    regressor = xgb.XGBRegressor(**params)
    regressor.fit(X,y)
    shap_calculations_xgb(regressor, X)


    return result_dict

    min_value = min(result_dict['mae'])
    max_value = max(result_dict['evs'])
    min_index = result_dict['mae'].index(min_value)
    max_index = result_dict['evs'].index(max_value)
    
    

    
    print(min_index, max_index, result_dict['fjoldi'][min_index])

def regression_over_possible_values_random_forest(X, y, interval_int):
    test_size = .25
    seed = 4
    result_dict = {'fjoldi': [], 'mae': [], 'rmse': [], 'r2': [], 'evs': []}
    

    forest = RandomForestRegressor(n_estimators=100,
        criterion='absolute_error',
        bootstrap = 'True',
        random_state=1,
        n_jobs=-1)
        
    for add_int in range(0, 200000000, interval_int):

        X_train, X_test, y_train, y_test = train_test_split(X.iloc[14:, :],
                                                    y.iloc[14:],
                                                    test_size=test_size,
                                                    random_state=seed)
            


        forest.fit(X_train, y_train)
        
        y_pred_test = forest.predict(X_test)
        
        result_dict['fjoldi'].append(y[52])
        result_dict['mae'].append(mean_absolute_error(y_test,
                                                  y_pred_test))
        result_dict['rmse'].append(math.sqrt(mean_squared_error(y_test,
                                                            y_pred_test)))
        result_dict['r2'].append(r2_score(y_test,
                                      y_pred_test))
        result_dict['evs'].append(explained_variance_score(y_test,
                                                           y_pred_test))
        y[50] += interval_int
        y[51] += interval_int
        y[52] += interval_int
    y[51] = 475000000
    y[52] = 435000000

    regressor = RandomForestRegressor(n_estimators=35,
            criterion='absolute_error',
            random_state=1,
            n_jobs=-1)
    regressor.fit(X, y)
    shap_calculations_rf(regressor, X)


    return result_dict

    min_value = min(result_dict['mae'])
    max_value = max(result_dict['evs'])
    min_index = result_dict['mae'].index(min_value)
    max_index = result_dict['evs'].index(max_value)
    return result_dict

def plot_result_range(result_dict, interval_int, fractile, regressor_type):
    
    result_dict['x'] = range(343, 543, int(interval_int/1e6))
    fig, ax = plt.subplots()
    sns.set(style='whitegrid',
            palette='pastel', )
    sns.lineplot(x='x',
                 y='evs',
                 data=result_dict,
                 color="red",
                 ax=ax)
    ax.set(xlabel = 'size of stock in millions',
           ylabel = 'explainable variance, red',
           title='school fractile:'+fractile+'\n'+'regressor type :' + regressor_type + '\n'+'1996-2022',
           ylim=(0,1))
    
    ax2 = ax.twinx()
    sns.lineplot(x='x',
                 y='mae',
                 data=result_dict,
                 color='blue',
                 markers=True, ax=ax2)
    ax2.set(ylabel = 'mean average error, blue')
    plt.show()


fractile = '096'
interval_int = 1000000

(X,y) = get_new_data(fractile)
print(X)
result_dict = regression_over_possible_values_XGB(X, y, interval_int)
regressor_type='rgb'
plot_result_range(result_dict, interval_int, fractile, regressor_type )
result_dict_xgb = regression_over_possible_values_random_forest(X, y, interval_int)
regressor_type = 'rf'
plot_result_range(result_dict_xgb, interval_int, fractile, regressor_type)
# Checking for prinipcal components
scaler = StandardScaler()
X_sca = scaler.fit_transform(X)


pca = PCA(n_components=18)
pca.fit(X_sca)
print((pca.explained_variance_ratio_))
print(pca.singular_values_)







