# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 12:23:26 y_

@author: jst
"""

import pandas as pd
import math
#import numpy as np
#import seaborn as sns
from matplotlib import pyplot as plt
#from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import shap

#np.seterr(divide='ignore', invalid='ignore')

path_str = 'R:\\Ráðgjöf\\Bláa hagkerfið\\Hafró\\distribution_output\\'
path_str_gr = 'R:\\Ráðgjöf\\Bláa hagkerfið\\Hafró\\golden_redfish_data\\'
path_str_mo = 'R:\\Ráðgjöf\\Maris Optimum/Golden_redfish_model\\'

SEASON_int = 5 #spring


XX_2022_dict={"lengd":[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60, 'catch'],
              "y":[0,0,1.73,2.796,3.405,2.969,3.761,2.299,2.299,1.805,1.191,1.724,1.724,1.724,2.24,2.299,2.299,1.737,2.21,2.299,2.299,2.902,3.448,3.448,3.448,2.874,2.939,3.448,3.448,7.011,15.238,40.273,68.759,74.023,78.7,51.832,37.509,28.046,20.581,12.794,8.02,5.445,3.815,2.938,2.38,0,0,0,0,0,0,0,0,0,0,0,0]}
sum_2022_y=sum(XX_2022_dict['y'])
MEASUREMENT_QTY_2022 = 104407 * 168/202
CATCH_2022 = 26000
XX_2022_df = pd.DataFrame.from_dict(XX_2022_dict) 
XX_2022_df['z']=XX_2022_df['y']*MEASUREMENT_QTY_2022/sum_2022_y
XX_2022_df.z.at[56]=-CATCH_2022
XX_2022_df.index=[0]*len(XX_2022_df)
XX_2022 = XX_2022_df.drop(columns='y').pivot(index=None,
                                             columns='lengd', 
                                             values='z')

X_df = pd.read_csv(path_str+'distribution.csv',
                   sep =",")
catch_df = pd.read_csv(path_str+'golden_redfish_catch.csv',
                       sep =";")
catch_df.at[2022]=26

X_cal_df = pd.read_csv(path_str_mo+'distribution_commercial.csv', 
                       sep=',')
X_cal_df = X_cal_df.pivot(index='ar', 
                          columns = 'lengd',
                          values = 'percent_per_year')
x_cal_df = X_cal_df.fillna(0)
X_cal_df.columns = 1000 + X_cal_df.columns
X_cal_df.index += 1
X_cal_df.columns = X_cal_df.columns.astype(int).astype(str)



XX_df=X_df.pivot(index='ar', 
                 columns='lengd', 
                 values='sum_fjoldi')

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
for i in range(5,60):
    XX_df.at[2022,str(i)]=XX_2022.loc[0,i]





XXH_2021_dict= {"x":[16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54],
 "y":[0.125,0.125,0.201,0.708,0.457,0.374,0.374,0.453,0.811,0.201,0.943,0.667,0.499,0.499,1.123,1.123,1.74,2.121,3.679,9.917,23.615,39.645,46.734,50.748,43.742,31.137,30.46,14.397,10.982,6.198,3.41,1.929,0.883,0.748,0.748,0.748,0.844,1.097,0.815]}

XXH_2021_df = pd.DataFrame.from_dict(XXH_2021_dict) 

XH_df =pd.read_csv(path_str+'distributionH.csv',
                   sep =",")

XXH_df=XH_df.pivot(index='ar', 
                   columns='lengd', 
                   values='sum_fjoldi')


XXH_df.columns = XX_df.columns.astype(int).astype(str)
XXH_df = XXH_df.drop(2011)
XXH_df = XXH_df.iloc[1:25,:]

YX = pd.read_csv(path_str+"RED_numbers_at_age.csv", sep=";")

YXH= pd.read_csv(path_str+"RED_smh.csv", sep=",")
YXH['sum']=YXH.iloc[:,3:29].sum(axis=1)*1e6
YXH=YXH['sum']

YY=YX.iloc[15:53,28]
#YY[51]=450000000
#YY[52]=400000000


XX_df['catch']=catch_df.catch.values*-1000
XX_df = XX_df.join(X_cal_df.iloc[:,:])

XX_df =  XX_df.fillna(0)



XXH_df['catch']=catch_df.catch.iloc[12:36].values*-1000
XXH_df.at[2020, 'catch']=-36000


XXH_df = XXH_df.join(X_cal_df.iloc[:,:])


XX_2021 = XX_df.loc[2021].to_frame().transpose()
XXH_2021 = XXH_df.loc[2020].to_frame().transpose()

### begin autumnal data
'''
XX_df=XXH_df
YY=YXH
XX_2021 =XXH_2021
YY=YX.iloc[:52,28]
SEASON_int = 32 #autumn
'''
### end autumnal data





parameters = {
    'nthread': [0],
    'objective': ['reg:squarederror'],
    'eval_metric': ["error"],
    'learning_rate': [0.4,0.5],
    'max_depth': [3, 4,5],
    'min_child_weight': [4],
    'subsample': [0.5],
    'colsample_bytree': [0.6],
    'n_estimators': [30]
}

'''
#parameters for autumn optimization.

parameters = {
    'nthread': [0],
    'objective': ['reg:squarederror'],
    'eval_metric': ["mlogloss"],
    'learning_rate': [0.4],
    'max_depth': [3,],
    'min_child_weight': [5],
    'subsample': [0.5,0.6,0.7      ],
    'colsample_bytree': [0.7, 0.8,0.9],
    'n_estimators': [1000]
}
'''

test_size = .25
seed = 4

result_dict = {'fjoldi':[],'mae':[],'rmse':[], 'r2':[],'evs':[]}

YY[50]-=20000000
YY[51]-=20000000




for add_int in range(0,160000000,5000000):



    X_train, X_test, y_train, y_test = train_test_split(XX_df.iloc[16:,:],
                                                        YY.iloc[16:],
                                                        test_size=test_size,
                                                        random_state=seed)
    '''
    X_train = XX_df.loc[:2010,:]
    X_test = XX_df.loc[2011:,:]
    y_train = YY[:26]
    y_test = YY[26:]
    
    '''
    
    
    xgb1 = xgb.XGBRegressor(seed)
    xgb_regressor = GridSearchCV(xgb1, 
                                 parameters, 
                                 cv=2, 
                                 n_jobs=5, 
                                 verbose=5)
    xgb_regressor.fit(XX_df, 
                      YY)
    print(xgb_regressor.best_score_)
    print(xgb_regressor.best_params_)
    
    params = xgb_regressor.best_params_
    eval_set = [(X_train, y_train), (X_test, y_test)]
    xgb_regressor = xgb.XGBRegressor(**params)
    
    xgb_regressor.fit(X_train,
                      y_train,
                      eval_metric=["mae"],
                      eval_set=eval_set,
                      verbose=False)
    
    y_pred_test = xgb_regressor.predict(X_test)
    
    y_pred_2021 = xgb_regressor.predict(XX_2021)
    #y_pred_2022 = xgb_regressor.predict(XX_df.loc[2022,:])
    
    
    print(YY[51])
    print("mae", mean_absolute_error(y_test, y_pred_test))
    print("rmse", math.sqrt(mean_squared_error(y_test, y_pred_test)))
    print("r2", r2_score(y_test, y_pred_test))
    print("evs", explained_variance_score(y_test, y_pred_test))

    result_dict['fjoldi'].append(YY[51])
    result_dict['mae'].append(mean_absolute_error(y_test, y_pred_test))
    result_dict['rmse'].append( math.sqrt(mean_squared_error(y_test, y_pred_test)))
    result_dict['r2'].append(r2_score(y_test,y_pred_test))
    result_dict['evs'].append(explained_variance_score(y_test, y_pred_test))
    
    YY[51] +=5000000
    YY[52] +=5000000                             




'''


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

results = xgb_regressor.evals_result()
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

#explainer = shap.TreeExplainer(xgb_regressor)



explainer = shap.TreeExplainer(xgb_regressor)

shap_values = explainer.shap_values(XX_df.iloc[16:,:])

shap.summary_plot(shap_values, 
                  XX_df.iloc[16:,:],
                  plot_type="layered_violin", 
                  max_display=10)

shap_values = explainer(XX_df.iloc[16:,:])
shap.plots.waterfall(shap_values[21])


X50 = shap.utils.sample(XX_df, 50)
explainer = shap.Explainer(xgb_regressor.predict, X50)
shap_values = explainer(XX_df)


sample_ind = 37
shap.partial_dependence_plot(
    "1035", xgb_regressor.predict, X50, model_expected_value=True,
    feature_expected_value=True, ice=False,
    shap_values=shap_values[sample_ind:sample_ind+1,:])



'''



