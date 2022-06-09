# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 12:23:26 y_

@author: jst
"""

import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
import xgboost as xgb
# import seaborn as sbn
from sklearn.model_selection import GridSearchCV

import shap


XX_2022_dict={"x":[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60, 'catch'],
              "y":[0,0,1.73,2.796,3.405,2.969,3.761,2.299,2.299,1.805,1.191,1.724,1.724,1.724,2.24,2.299,2.299,1.737,2.21,2.299,2.299,2.902,3.448,3.448,3.448,2.874,2.939,3.448,3.448,7.011,15.238,40.273,68.759,74.023,78.7,51.832,37.509,28.046,20.581,12.794,8.02,5.445,3.815,2.938,2.38,0,0,0,0,0,0,0,0,0,0,0,0]}
sum_2022_y=sum(XX_2022_dict['y'])
MEASUREMENT_QTY_2022 = 145983 * 168755 / 209102
CATCH_2022 = 34000
XX_2022_df = pd.DataFrame.from_dict(XX_2022_dict) 
XX_2022_df['z']=XX_2022_df['y']*MEASUREMENT_QTY_2022/sum_2022_y
XX_2022_df.z.at[56]=-CATCH_2022
XX_2022_df.index=[0]*len(XX_2022_df)
XX_2022 = XX_2022_df.drop(columns='y').pivot(index=None,columns='x', values='z')



XXH_2021_dict= {"x":[16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54],
 "y":[0.125,0.125,0.201,0.708,0.457,0.374,0.374,0.453,0.811,0.201,0.943,0.667,0.499,0.499,1.123,1.123,1.74,2.121,3.679,9.917,23.615,39.645,46.734,50.748,43.742,31.137,30.46,14.397,10.982,6.198,3.41,1.929,0.883,0.748,0.748,0.748,0.844,1.097,0.815]}



XXH_2021_df = pd.DataFrame.from_dict(XXH_2021_dict) 




path_str = 'R:\\Ráðgjöf\\Bláa hagkerfið\\Hafró\\distribution_output\\'
path_str_gr = 'R:\\Ráðgjöf\\Bláa hagkerfið\\Hafró\\golden_redfish_data\\'



X_df = pd.read_csv(path_str+'distribution.csv',sep =",")
XH_df =pd.read_csv(path_str+'distributionH.csv',sep =",")


XX_df=X_df.pivot(index='ar', columns='lengd', values='sum_fjoldi')
XXH_df=XH_df.pivot(index='ar', columns='lengd', values='sum_fjoldi')


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



XXH_df.columns = XX_df.columns.astype(int).astype(str)
XX_df.fillna(0, inplace=True)     





XXH_df.fillna(0, inplace=True)   
XXH_df.drop(2011,inplace=True)
XXH_df = XXH_df.iloc[1:25,:]

YX = pd.read_csv(path_str+"RED_numbers_at_age.csv", sep=";")

YXH= pd.read_csv(path_str+"RED_smh.csv", sep=",")
YXH['sum']=YXH.iloc[:,3:29].sum(axis=1)*1e6
YXH=YXH['sum']


YY=YX.iloc[15:52,28]



catch_df = pd.read_csv(path_str+'golden_redfish_catch.csv',sep =";")

XX_df['catch']=catch_df.catch.values*-1000
XXH_df['catch']=catch_df.catch.iloc[12:36].values*-1000
XXH_df.at[2020, 'catch']=-36000 #forcing this number




STARTING_YEAR = 1985
ENDING_YEAR = 2021
old_year= STARTING_YEAR



test_size = .35
seed = 2

XX_2021 = XX_df.loc[2021].to_frame().transpose()
XXH_2021 = XXH_df.loc[2020].to_frame().transpose()

"""


XX_df=XXH_df
YY=YXH
XX_2021 =XXH_2021

YY=YX.iloc[:52,28]




"""





XX_trainR, XX_testR, yY_trainR, yY_testR = train_test_split(XX_df,
                                                    YY,
                                                    test_size=test_size,
                                                    random_state=seed)


scaler = StandardScaler().fit(XX_trainR)
X_train_norm = scaler.transform(XX_trainR)
X_norm = scaler.transform(XX_df)
X_test_norm = scaler.transform(XX_testR)

xgb1 = xgb.XGBRegressor(seed=2)

parameters = {
    'nthread': [0],
    'objective': ['reg:squarederror'],
    'eval_metric': ["error"],
    'learning_rate': [ 0.3 ,0.4, 0.5, 0.6],
    'max_depth': [3,6, 7],
    'min_child_weight': [1, 3, 5, 7],
    'subsample': [0.9, 0.7, 0.6],
    'colsample_bytree': [0.7, 0.8,0.9],
    'n_estimators': [100, 200, 300]
}

xgb_regressor = GridSearchCV(xgb1, parameters, cv=2, n_jobs=5, verbose=5)

xgb_regressor.fit(XX_df, YY)


print(xgb_regressor.best_score_)

print(xgb_regressor.best_params_)

params = xgb_regressor.best_params_
eval_set = [(XX_trainR, yY_trainR), (XX_testR, yY_testR)]
xgb_regressor = xgb.XGBRegressor(**params)

xgb_regressor.fit(XX_trainR,
                  yY_trainR,
                  eval_metric=["mae"],
                  eval_set=eval_set,
                  verbose=False)



y_pred_test = xgb_regressor.predict(XX_testR)


y_pred_2021 = xgb_regressor.predict(XX_2021)
y_pred_2022 = xgb_regressor.predict(XX_2022)




print("mae", mean_absolute_error(yY_testR, y_pred_test))
print("rmse", math.sqrt(mean_squared_error(yY_testR, y_pred_test)))
print("r2", r2_score(yY_testR, y_pred_test))
print("evs", explained_variance_score(yY_testR, y_pred_test))




x_ax = range(len(yY_testR))
plt.scatter(x_ax, yY_testR, s=5, color="blue", label="original")
plt.plot(x_ax, y_pred_test, lw=0.8, color="red", label="predicted")
plt.grid(True)
plt.legend()
plt.show()

predictions = [round(value) for value in y_pred_test]



"""

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


"""





#explainer = shap.TreeExplainer(xgb_regressor)

explainer = shap.Explainer(xgb_regressor,XX_df)

shap_values = explainer.shap_values(XX_df)



shap.summary_plot(shap_values, XX_df,plot_type="layered_violin", )



shap_values=explainer(XX_df)


shap.plots.waterfall(shap_values[36], max_display=10)

XX_df

"""

import matplotlib.ticker as ticker

x=XX_df.columns[0:56]
y=XX_df.iloc[23,0:56]

ax  =sbn.barplot(y,x,orient='h', dodge= False )

ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter(5))

plt.show()

"""


