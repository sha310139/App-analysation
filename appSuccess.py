import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
sns.set(font_scale=1.8)

import plotly.offline as py
from plotly import tools
#py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import missingno as msno
import random

from plotly import tools


# 讀入資料集

df_app = pd.read_csv('AppleStore.csv')

print(df_app.isnull().sum())  # df_app中沒有null的資料

# 把bytes轉成MB, 方便查看跟處理
df_app['size_bytes_in_MB'] = df_app['size_bytes'] / (1024 * 1024.0)

# 根據價錢增加新的項目 : 是否需付費
df_app['isNotFree'] = df_app['price'].apply(lambda x: 1 if x > 0 else 0)

df_app = df_app.iloc[:, 1:]

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb


df_app['rating_count_before'] = df_app['rating_count_tot'] - df_app['rating_count_ver']

df_train = df_app[['size_bytes_in_MB', 'isNotFree', 'price', 'rating_count_before', 'ipadSc_urls.num', 'lang.num', 'vpp_lic', 'prime_genre']]
#df_train = df_app[['isNotFree', 'ipadSc_urls.num', 'lang.num', 'prime_genre']]
target = df_app['user_rating']

df_train = pd.get_dummies(df_train)

def categorize_rating(x):
    if x <= 4:
        return 0
    else:
        return 1

target = target.apply(categorize_rating)

target.astype(str).hist()



X_train, X_test, y_train, y_test = train_test_split(df_train.values, target, test_size=0.4, random_state=3000, stratify=target)


#print('X_train shape:', X_train.shape)
#print('X_test shape:', X_test.shape)



from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_validate





from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

from sklearn.svm import SVC



#gsearch1 = GridSearchCV(estimator = lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',metrics='auc',learning_rate=0.1, n_estimators=188, max_depth=6, bagging_fraction = 0.8,feature_fraction = 0.8), 


models = [SVC(gamma='auto'), 
          DecisionTreeClassifier(max_depth=4), 
          RandomForestClassifier(n_estimators= 60, max_depth=20, max_leaf_nodes=20), 
          LGBMClassifier(boosting_type='gbdt',objective='binary',metrics='auc',learning_rate=0.1, max_depth=5, num_leaves=30, n_estimators=188,subsample=0.8, bagging_fraction = 0.8,feature_fraction = 0.8), 
          XGBClassifier(learning_rate =0.1,n_estimators=1000,max_depth=5,min_child_weight=1,gamma=0,subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=27)]

kfold = KFold(n_splits=10, random_state=3000, shuffle=True)

clf_comparison = pd.DataFrame(columns=['Classfier_name', 'train_score', 'test_score'])

for i, model in enumerate(models):
    clf = model
    clf.fit(X_train, y_train)
    cv_result = cross_validate(model, df_train, target, cv=kfold, scoring='accuracy', return_train_score=True)
    clf_comparison.loc[i, 'Classfier_name'] = model.__class__.__name__
    clf_comparison.loc[i, 'train_score'] = cv_result['train_score'].mean()
    clf_comparison.loc[i, 'test_score'] = cv_result['test_score'].mean()

    print(clf.score(X_test, y_test))


print(clf_comparison)


