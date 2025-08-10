import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv("C:\\Users\\knox0\\Desktop\\Proje 2\\Veriler\\house-prices-advanced-regression-techniques\\train.csv")



print(df.head())
df=df.drop("Id",axis =1)
print(df.describe())

print(df["PoolQC"].isnull().sum())
print(df["PoolQC"].value_counts())
df=df.drop("PoolQC",axis=1)




print(df["MiscFeature"].isnull().sum())
print(df["MiscFeature"].value_counts())
df=df.drop("MiscFeature",axis=1)


print(df["Fence"].isnull().sum())
df=df.drop("Fence",axis=1)

df=df.drop("Alley",axis=1)

print(df[df['Electrical'].isna()])
df = df.drop(index=1379)

print(df.head())
print(df.describe())

df=df.drop("MasVnrType",axis=1)

df=df.drop("FireplaceQu",axis=1)


print(df.head())
print(df.describe())



num_fill = ["LotFrontage","MasVnrArea","GarageYrBlt"]

for i in num_fill:
    df[i].fillna(df[i].median(),inplace=True)


obj_fill = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2","GarageType", "GarageFinish", "GarageQual", "GarageCond" ]

for i in obj_fill:
    df[i].fillna("None",inplace=True)


print(df.info())

df[df.isnull().any(axis=1)]


X = df.drop(["SalePrice"],axis=1)
y = df["SalePrice"]


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=42)




cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

column_trans = ColumnTransformer(

    transformers=[

        ("cat",OneHotEncoder(drop="first",sparse_output=False,handle_unknown="ignore"),cat_cols)
    ],remainder='passthrough'
)


X_train_encoded = column_trans.fit_transform(X_train)
X_test_encoded= column_trans.transform(X_test)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)


from sklearn.metrics import mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()

from sklearn.model_selection import RandomizedSearchCV

param ={

    "n_estimators" :[100,200,300],
    "max_depth" : [10,15,20],
    "min_samples_split" : [5,8,10],
    "min_samples_leaf" : [2,3,4],
    "max_features" :["sqrt","log2"],
    "bootstrap" : [True]

}

random = RandomizedSearchCV(estimator=rf,param_distributions=param,cv=5,n_iter=30,n_jobs=-1,random_state=42)

random.fit(X_train_scaled,y_train)
best_est_random = random.best_estimator_

y_pred_random =best_est_random.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred_random)
r2 = r2_score(y_test, y_pred_random)

print("mean score for rf \n",mse)
print("r2 score for rf \n", r2)


import xgboost as xgb


xgb_model = xgb.XGBRegressor()

xgb_param = {

    "n_estimators": [100,200,300],
    "learning_rate" : [0.01,0.05, 0.1],
    "max_depth" : [3,5,7,10],
    "min_child_weight" : [3,5,7,10],
    "subsample" : [0.6, 0.8,0.9],
    "colsample_bytree" : [0.6, 0.8, 0.9],
    "gamma" : [0.1, 0.2, 0.4, 0.6, 1]

}
random = RandomizedSearchCV(estimator=xgb_model,param_distributions=xgb_param,cv=5,n_iter=30,n_jobs=-1,random_state=42)


random.fit(X_train_scaled,y_train)
best_est_random = random.best_estimator_

y_pred_xgb =best_est_random.predict(X_test_scaled)

print("XGBoost MSE:", mean_squared_error(y_test, y_pred_xgb))
print("XGBoost R2:", r2_score(y_test, y_pred_xgb))


from catboost import CatBoostRegressor

cat_model = CatBoostRegressor(random_state=42, verbose=0)

param_dist = {
    "depth": [4, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "iterations": [100, 200, 500],
    "l2_leaf_reg": [1, 3, 5, 7, 9],
    "border_count": [32, 50, 100],
    "bagging_temperature": [0, 1, 3, 5]
}

random_search = RandomizedSearchCV(cat_model, param_distributions=param_dist,
                                   n_iter=30, cv=3, verbose=2, n_jobs=-1,
                                   random_state=42)

random_search.fit(X_train_scaled, y_train)

print("Best CatBoost params:", random_search.best_params_)

best_cat = random_search.best_estimator_

y_pred_cat = best_cat.predict(X_test_scaled)

print("CatBoost MSE:", mean_squared_error(y_test, y_pred_cat))
print("CatBoost R2:", r2_score(y_test, y_pred_cat))


from sklearn.linear_model import Ridge


ridge = Ridge(random_state=42)

param_dist = {
     "alpha": np.linspace(0.1, 10, 30),
    "solver": ["auto", "svd", "cholesky", "lsqr"],
    "fit_intercept": [True, False]
}

random_search = RandomizedSearchCV(
    estimator=ridge,
    param_distributions=param_dist,
    n_iter=30,  # Denenecek rastgele parametre sayısı
    cv=5,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train_scaled, y_train)

print("Best Ridge params:", random_search.best_params_)

best_ridge = random_search.best_estimator_

y_pred_ridge = best_ridge.predict(X_test_scaled)

print("Ridge MSE:", mean_squared_error(y_test, y_pred_ridge))
print("Ridge R2:", r2_score(y_test, y_pred_ridge))


