# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 12:23:26 2021

@author: jst
"""


import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import shap


path_str = 'R:\\Ráðgjöf\\Bláa hagkerfið\\Hafró\\distribution_output\\'
path_str_gr = 'R:\\Ráðgjöf\\Bláa hagkerfið\\Hafró\\golden_redfish_data\\'








X_df = pd.read_csv(path_str+'distribution.csv',sep =",")
XX_df=X_df.pivot(index='ar', columns='lengd', values='sum_fjoldi')

# Clean datasets
XX_df.drop(6.4,axis=1, inplace=True)
XX_df.drop(11.9,axis=1, inplace=True)
XX_df.drop(12.5,axis=1, inplace=True)
XX_df.drop(12.6,axis=1, inplace=True)
XX_df.drop(13.1,axis=1, inplace=True)
XX_df.drop(13.4,axis=1, inplace=True)
XX_df.drop(13.6,axis=1, inplace=True)
XX_df.drop(13.7,axis=1, inplace=True)
XX_df.drop(13.9,axis=1, inplace=True)
XX_df.drop(14.4,axis=1, inplace=True)
XX_df.drop(14.5,axis=1, inplace=True)
XX_df.drop(14.7,axis=1, inplace=True)
XX_df.drop(14.8,axis=1, inplace=True)
XX_df.drop(14.9,axis=1, inplace=True)


YX = pd.read_csv(path_str+"RED_numbers_at_age.csv", sep=";")
XX_df.fillna(0, inplace=True)     



catch_df = pd.read_csv(path_str+'golden_redfish_catch.csv',sep =";")





def find_per(year, lengd):
    for index, row in X_df.iterrows():
        if row[0]==year and row[1]==lengd: 
            return row[5]



STARTING_YEAR = 1985
ENDING_YEAR = 2021
old_year= STARTING_YEAR

for index, row in X_df.iterrows():
    new_year = int(row[0])
    new_lengd = row[1]
    new_per = row[5]
    if old_year != new_year:
        try:
            old_per = find_per(new_year-1, new_lengd-1)
            X_df.at[index,'diff']= old_per - new_per
        except:
            pass
        
        
X_df.fillna(0, inplace=True)        
            
X1_df =X_df                  
X1_df.drop(X1_df[X1_df.ar == 1985].index, inplace=True)        
        
            


X = X1_df.loc[:,['ar','lengd']]

Y1 = X1_df.loc[:,['diff']]

for index, row in X.iterrows():
    X.at[index,'afli']=catch_df.iloc[int(row[0]-1986),1]



test_size = .2
seed = 0 

line_int=0
X_test1=pd.DataFrame()

for length in range(5,61):
    X_test1.at[line_int,1]=int(2023)
    X_test1.at[line_int,2]=int(length) 
    X_test1.at[line_int,3]=int(72) 
    
    line_int+=1
    
X_test1.columns=['ar','lengd', 'afli']

X_trainR, X_testR, y_trainR, y_testR = train_test_split(X,
                                                    Y1,
                                                    test_size=test_size,
                                                    random_state=seed)





#mean_train = np.mean(y_trainR)
#baseline_predictions = np.ones(y_testR.shape) * mean_train
# Reikna meðal skekkju

#mae_baseline = mean_absolute_error(y_testR, baseline_predictions)
#print("Baseline MAE is {:.3f}".format(mae_baseline))

xgb1 = xgb.XGBRegressor(seed=2)

parameters = {
    'nthread': [0],
    'objective': ['reg:squarederror'],
    'eval_metric': ["error"],
    'learning_rate': [0.01, 0.1, 0.05],
    'max_depth': [6],
    'min_child_weight': [1, 3, 5, 7],
    'subsample': [0.9],
    'colsample_bytree': [0.7],
    'n_estimators': [100, 200, 300, 1000]
}

xgb_regressor = GridSearchCV(xgb1, parameters, cv=2, n_jobs=5, verbose=5)

xgb_regressor.fit(X, Y1)


print(xgb_regressor.best_score_)

print(xgb_regressor.best_params_)

y_pred = xgb_regressor.predict(X_testR)
y_pred_xgb = xgb_regressor.predict(X_test1)


per_2021_lst=[]
for index, row in X_df.iterrows(): 
    if row[0] == 2021:
        per_2021_lst.append(row[5])


y_pred_xgb=y_pred_xgb+per_2021_lst



print("mae", mean_absolute_error(y_testR, y_pred))
print("rmse", math.sqrt(mean_squared_error(y_testR, y_pred)))
print("r2", r2_score(y_testR, y_pred))
print("evs", explained_variance_score(y_testR, y_pred))




#y_pred_svm = gs.predict(X_test1)
predictions = [round(value) for value in y_pred]

Forecast = np.vstack((y_pred_xgb))
Forecast_df = pd.DataFrame(Forecast, columns=["xgb"], index=X_test1.index)
Forecast_df.reset_index()

params = xgb_regressor.best_params_
xgb_regressor = xgb.XGBRegressor(**params)
eval_set = [(X_trainR, y_trainR), (X_testR, y_testR)]
xgb_regressor.fit(X_trainR,
                  y_trainR,
                  eval_metric=["mae"],
                  eval_set=eval_set,
                  verbose=False)

results = xgb_regressor.evals_result()
epochs = len(results['validation_0']['mae'])
x_axis = range(0, epochs)


# plot classification error
fig, ax = plt.subplots()
plt.grid(True, which='major')
ax.plot(x_axis, results['validation_0']['mae'], label='Train')
ax.plot(x_axis, results['validation_1']['mae'], label='Test')
ax.legend()
plt.ylabel('Error')
plt.title('Error')
plt.show()





svr_regr = make_pipeline(StandardScaler(), SVR(kernel='rbf',
                                               C=1.0,
                                               epsilon=0.2))
svr_regr.fit(X, Y1)
y_pred_svr = svr_regr.predict(X_test1)

Forecast = np.vstack((y_pred_xgb, y_pred_svr))
Forecast_df = pd.DataFrame(Forecast,
                           columns=X_test1.index,
                           index=['xgb', 'svr']).transpose()

Forecast_df['xgb'] = pd.Series(
    ["{0:.2f}%".format(val * 100) for val in Forecast_df['xgb']],
    index=Forecast_df.index)
Forecast_df['svr'] = pd.Series(
    ["{0:.2f}%".format(val * 100) for val in Forecast_df['svr']],
    index=Forecast_df.index)
Forecast_df

Forecast_df['ar']=2023

length_df=pd.DataFrame(np.arange(5,61))
length_df.columns=['lengd']

Forecast_df.drop('svr', axis=1, inplace=True)
Forecast_df.columns=['per','ar']
Forecast_df['per'] = Forecast_df['per'].str.rstrip('%').astype('float') / 100.0
Forecast_df=pd.concat([Forecast_df,length_df['lengd']],axis=1)
result_df=pd.concat([X_df['ar'],X_df['lengd'],X_df['per']], axis=1)

result_df=result_df.append(Forecast_df, ignore_index=True)


result_df.to_csv(path_str+'distribution_forecast.csv')


explainer = shap.TreeExplainer(xgb_regressor)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X)


idx = 10

shap.force_plot(explainer.expected_value, 
                shap_values[idx], 
                X_test1.iloc[idx,:]) 


