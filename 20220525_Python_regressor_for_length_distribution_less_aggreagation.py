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
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

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

XX_df.columns = XX_df.columns.astype(int).astype(str)


YX = pd.read_csv(path_str+"RED_numbers_at_age.csv", sep=";")
YY=YX.iloc[15:52,28]
XX_df.fillna(0, inplace=True)     



catch_df = pd.read_csv(path_str+'golden_redfish_catch.csv',sep =";")

XX_df['catch']=catch_df.catch.values



def find_per(year, lengd):
    for index, row in X_df.iterrows():
        if row[0]==year and row[1]==lengd: 
            return row[5]



STARTING_YEAR = 1985
ENDING_YEAR = 2021
old_year= STARTING_YEAR



test_size = .3
seed = 2




XX_trainR, XX_testR, yY_trainR, yY_testR = train_test_split(XX_df,
                                                    YY,
                                                    test_size=test_size,
                                                    random_state=seed)

XX_2021 = XX_df.loc[2021].to_frame().transpose()

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

xgb_regressor.fit(XX_df, YY)


print(xgb_regressor.best_score_)

print(xgb_regressor.best_params_)

y_pred = xgb_regressor.predict(XX_testR)


y_pred_xgb = xgb_regressor.predict(XX_2021)



print("mae", mean_absolute_error(yY_testR, y_pred))
print("rmse", math.sqrt(mean_squared_error(yY_testR, y_pred)))
print("r2", r2_score(yY_testR, y_pred))
print("evs", explained_variance_score(yY_testR, y_pred))




predictions = [round(value) for value in y_pred]

Forecast = np.vstack((y_pred_xgb))

params = xgb_regressor.best_params_


xgb_regressor = xgb.XGBRegressor(**params)
eval_set = [(XX_trainR, yY_trainR), (XX_testR, yY_testR)]

xgb_regressor.fit(XX_trainR,
                  yY_trainR,
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








#explainer = shap.TreeExplainer(xgb_regressor)
#"shap_values = explainer.shap_values(XX_df)

#shap.summary_plot(shap_values, XX_df,plot_type="layered_violin")


explainer = shap.Explainer(xgb_regressor,XX_df)
shap_values=explainer(XX_df)


shap.plots.waterfall(shap_values[36], max_display=520)

shap.summary_plot(shap_values, XX_df,plot_type="layered_violin")

