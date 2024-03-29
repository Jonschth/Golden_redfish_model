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
# from sklearn.pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import shap

def get_new_data(fractile):
    path_str = 'R:\\Ráðgjöf\\Bláa hagkerfið\\Hafró\\distribution_output\\'
    path_str_do = 'R:\\Ráðgjöf\\Maris Optimum/distribution_output\\'
    X_df = pd.read_csv(path_str_do+'distribution'+fractile+'.csv',
                   sep=",")
    
    ysq_df = X_df[['ar', 'max(cum)']]
    ysq_df.set_index(['ar'], inplace = True)
    ysq_df = ysq_df[~ysq_df.index.duplicated(keep='first')]
    
    catch_df = pd.read_csv(path_str+'golden_redfish_catch.csv',
                           sep=";")

    catch_df.at[37, 'year'] = 2022.0
    catch_df.at[37, 'catch'] = 26
    catch_df.at[37, 'number'] = 29 
    

    
    X_cal_df = pd.read_csv(path_str_do+'distribution_commercial.csv',
                           sep=",")
    X_cal_df.drop(1605, axis = 0, inplace = True)
    
    X_cal_df = X_cal_df.pivot(index='ar',
                              columns='lengd',
                              values='per_length')
    X_cal_df = X_cal_df.fillna(0)
    X_cal_df.columns = 1000 + X_cal_df.columns
    X_cal_df.columns = X_cal_df.columns.astype(int).astype(str)
    
    catch_df = catch_converter(X_cal_df, catch_df)
    catch_df.year = catch_df.year.astype(int)
    catch_df.set_index(catch_df.year, inplace = True)
    
    X_cal_df = X_cal_df.mul(catch_df.number*-1e6,axis = 0)
    
    XX_df = X_df.pivot(index='ar',
                       columns='lengd',
                       values='per_length')
    
    XX_df = pd.merge(XX_df, ysq_df, right_index = True, left_index = True)
    
    #XX_df.drop(4.5, axis=1, inplace=True)
    #XX_df.drop(6.4, axis=1, inplace=True)
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
    
    XX_df.columns = XX_df.columns.astype(str)

    YX = pd.read_csv(path_str+"RED_numbers_at_age.csv", sep=";")
    YY = YX.iloc[15:53, 28]
    
  
    
    XX_df = XX_df.join(X_cal_df.iloc[:, :])

    XX_df.index =XX_df.index.astype(str)  
    
    s = XX_df.index[27:35]
    s.index = XX_df[27:35]
    s = pd.get_dummies(s)
    s.index = XX_df.index[27:35]
    XX_df = XX_df.join(s)
    XX_df = XX_df.fillna(0)
    
    return (XX_df, YY)

def catch_converter(X_catch_per_df, catch_df ):
    from sklearn.linear_model import LinearRegression
    path_grs_str = 'R:/Ráðgjöf/Maris Optimum/Golden_redfish_model/'
    wl_df = pd.read_csv(path_grs_str+'RED_gadget_n_at_age.csv',sep=',')
    print(wl_df)
    Xl = wl_df[['year','mean_length']]
    yl = wl_df[['year','mean_weight']]
    for index,row in Xl.iterrows():
        Xl.at[index,'squared'] = row[1]**2
    for year in range(1985,2022):
        average_weight = 0
        reg_X = Xl[Xl['year'] == year]
        reg_y = yl[yl['year'] == year]
        reg = LinearRegression().fit(reg_X[['mean_length', 'squared']], reg_y[['mean_weight']])
        b = reg.coef_[0][0]
        a = reg.coef_[0][1]
        c = reg.intercept_
        print(year,a,b,c)
        for col in range(1010,1060):
            average_weight += ( X_catch_per_df.loc[year, str(col)])* (a*(col - 1000)**2 + b*(col - 1000) + c)
        catch_df.at[year - 1985, 'number'] = catch_df.loc[year - 1985,'catch']/average_weight
    return catch_df
    


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
    
    shap_values = explainer.shap_values(XX_df)
    
    shap.summary_plot(shap_values,
                      XX_df,
                      plot_type="bar",
                      max_display=40)
    
    shap_values = explainer(XX_df)
    shap.waterfall_plot(shap_values[21], max_display=40)
    shap.waterfall_plot(shap_values[19], max_display=40)
    shap.waterfall_plot(shap_values[17], max_display=40)
    shap.waterfall_plot(shap_values[15], max_display=40)
    shap.waterfall_plot(shap_values[13], max_display=40)
    

def shap_calculations_rf(regressor,XX_df):
        
    X = XX_df.iloc[16:,:]
    explainer = shap.TreeExplainer(regressor)
    
    shap_values = explainer.shap_values(X)
    
    shap.summary_plot(shap_values,
                      X,
                      plot_type="layered_violin",
                      max_display=30)
    
    class helper_object():
        def __init__(self, i):
            self.base_values = shap_values.base_values[i][0]
            self.data = shap_values.data[i]
            self.feature_names = X.columns.to_list()
            self.values = shap_values.values[i]
    shap_values = explainer(X)
    
    shap.waterfall_plot(helper_object(37))
    shap.waterfall_plot(helper_object(35))
    shap.waterfall_plot(helper_object(33))
    shap.waterfall_plot(helper_object(31))
    

def regression_over_possible_values_XGB(X,y, interval_int):
    parameters = {
        'nthread': [0],
        'objective': ['reg:squarederror'],
        'eval_metric': ["mae"],
        'learning_rate': [.2, .3, .4,.5,.7, .9],
        'max_depth': [1, 2, 3, 4, 5],
        'min_child_weight': [1, 2, 3, 4, 5],
        'subsample': [0.5, 0.6, 0.7],
        'colsample_bytree': [.5, .6, .7],
        'n_estimators': [55]
    }
    test_size = .25
    seed = 4
    result_dict = {'fjoldi2022': [], 'fjoldi2021': [], 'fjoldi2020': [], 'mae': [], 'rmse': [], 'r2': [], 'evs': []}

    xgb1 = xgb.XGBRegressor(seed)
    
    for add_int in range(0, 5000000, interval_int):
    
        X_train, X_test, y_train, y_test = train_test_split(X.iloc[14:, :],
                                                            y.iloc[14:],
                                                            test_size=test_size,
                                                            random_state=seed)
    

        xgb_regressor = RandomizedSearchCV(xgb1,
                                     parameters,
                                     cv=2,
                                     n_iter = 50,
                                     verbose=0)
    

        eval_set = [(X_train, y_train), (X_test, y_test)]

    
        xgb_regressor.fit(X_train,
                          y_train,
                          eval_set=eval_set,
                          verbose=False)
    
        y_pred_test = xgb_regressor.predict(X_test)

    
        result_dict['fjoldi2022'].append(y[52])
        result_dict['fjoldi2021'].append(y[51])
        result_dict['fjoldi2020'].append(y[50])

        result_dict['mae'].append(mean_absolute_error(y_test,
                                                      y_pred_test))
        result_dict['rmse'].append(math.sqrt(mean_squared_error(y_test,
                                                                y_pred_test)))
        result_dict['r2'].append(r2_score(y_test,
                                          y_pred_test))
        result_dict['evs'].append(explained_variance_score(y_test,
                                                           y_pred_test))
        y[50] += interval_int * (y[50]/y[51])
        y[51] += interval_int *(y[51]/y[52])
        y[52] += interval_int
    
    min_value = min(result_dict['mae'])
    min_index = result_dict['mae'].index(min_value)
 
    
    
    y[50] = result_dict['fjoldi2020'][min_index]
    y[51] = result_dict['fjoldi2021'][min_index]
    y[52] = result_dict['fjoldi2022'][min_index]
    
    regressor = GridSearchCV(xgb1,
                                     parameters,
                                     cv=3,
                                     n_jobs=50,
                                     verbose=0)
    
    regressor.fit(X, y)
    params = regressor.best_params_
    regressor = xgb.XGBRegressor(**params)
    regressor.fit(X,y)
    shap_calculations_xgb(regressor, X)


    return result_dict


def regression_over_possible_values_random_forest(X, y, interval_int):
    test_size = .25
    seed = 4
    result_dict = {'fjoldi': [], 'mae': [], 'rmse': [], 'r2': [], 'evs': []}
    

    forest = RandomForestRegressor(n_estimators=100,
        criterion='mae',
        bootstrap = 'True',
        random_state=1,
        n_jobs=-1)
        
    for add_int in range(0, 5000000, interval_int):

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
        y[50] += interval_int * (y[50]/y[51])
        y[51] += interval_int *(y[51]/y[52])
        y[52] += interval_int

    return result_dict



def plot_result_range(result_dict, interval_int, fractile, regressor_type):
    
    result_dict['x'] = range(343, 348, int(interval_int/1e6))
    fig, ax = plt.subplots()
    sns.set(style='whitegrid',
            palette='pastel', )
    sns.lineplot(x='x',
                 y='r2',
                 data=result_dict,
                 color="red",
                 ax=ax)
    ax.set(xlabel = 'size of stock in millions',
           ylabel = 'r2, red',
           title='school fractile:'+fractile+'\n'+'regressor type :' + regressor_type + '\n'+'1985-2022',
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

result_dict_gb = regression_over_possible_values_XGB(X, y, interval_int)
regressor_type='rgb'
plot_result_range(result_dict_gb, interval_int, fractile, regressor_type )

(X,y) = get_new_data(fractile)
result_dict_RF = regression_over_possible_values_random_forest(X, y, interval_int)
regressor_type = 'rf'
plot_result_range(result_dict_RF, interval_int, fractile, regressor_type)



'''
scaler = StandardScaler()
X_sca = scaler.fit_transform(X)
pca = PCA(n_components=18)
pca.fit(X_sca)
print((pca.explained_variance_ratio_))
print(pca.singular_values_)



y[43] -=7000000
y[44] -=8000000
y[45] -=8000000
y[46] -=7000000
y[47] -=4000000
y[48] -=4000000
y[49] -=1000000

'''

