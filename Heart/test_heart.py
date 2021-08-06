import random
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from xgboost import plot_importance, XGBRFClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import cross_val_score
from catboost import CatBoostClassifier
from sklearn import metrics
# To balance the dataset
from imblearn.over_sampling import SMOTE
from collections import Counter

# Start
# import data
df = pd.read_csv(r'../heart.csv', skip_blank_lines=True)
df = pd.DataFrame(df)
df = df.dropna()

# split
features = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall'
            ]
dataset = df[features]
result = df['output']
y = result.values
X = df[features].values

xgboostModel = XGBRFClassifier(colsample_bytree=.7, learning_rate=.05, max_depth=6, min_child_weight=11, missing=-999,
                               n_estimators=5, objective='binary:logistic', subsample=.8, nthread=4, seed=1337, silent=1
                               )
# data imbalance
over = SMOTE()
X, y = over.fit_resample(X, y)

counter = Counter(y)
print('The number of 0,1 : ', counter)

# Standardize or Normalize
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# PCA
# X = pca.fit_transform(X)
# T-SNE (Less data)
T_SNE = TSNE(n_components=2)
X = T_SNE.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=77)

# xgboost search CV
parameters = {'nthread': [4], # when use hyperthread, xgboost may become slower
              'objective': ['binary:logistic'],
              'learning_rate': [0.05], # so called `eta` value
              'max_depth': [6],
              'min_child_weight': [11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [5], # number of trees, change it to 1000 for better results
              'missing': [-999],
              'seed': [1337]}


clf = GridSearchCV(xgboostModel, parameters, n_jobs=5,
                   cv=StratifiedKFold(shuffle=True),
                   scoring='roc_auc',
                   verbose=2, refit=True)

clf.fit(X_train, y_train)
print(clf.best_params_)
# best_parameters, score, _ = max(clf.cv_results_, key=lambda x: x[1])
# print('Raw AUC score:', score)
# for param_name in sorted(best_parameters.keys()):
#    print("%s: %r" % (param_name, best_parameters[param_name]))

xgboostModel.fit(X_train, y_train)

predicted1 = xgboostModel.predict(X_test)

print('The score new XGBoost(train) is : ', xgboostModel.score(X_train, y_train))
print('The score new XGBoost(test) is : ', xgboostModel.score(X_test, y_test))

df_test = pd.DataFrame(X_test, columns=['PC1', 'PC2'])
df_test['Index'] = y_test
df_test['prediction'] = predicted1

sns.lmplot('PC1', 'PC2', hue='Index', data=df_test, fit_reg=False)
plt.title('new XGBoost (test--real)')
plt.show()
sns.lmplot('PC1', 'PC2', hue='prediction', data=df_test, fit_reg=False)
plt.title('new XGBoost (test--prediction)')
plt.show()

# accuracy score
print('-----------------classification_report of xgboost-------------------')
print(metrics.classification_report(y_test, predicted1))

sns.heatmap(metrics.confusion_matrix(y_test, predicted1), annot=True)
plt.title("Confusion Matrix 1")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# end

# xgboost
clf = XGBRFClassifier()
param_grid = {
        'silent': [False],
        'max_depth': [6, 10, 15, 20],
        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0,3],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
        'gamma': [0, 0.25, 0.5, 1.0],
        'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
        'n_estimators': [100]}

params = {'eval_metric': 'mlogloss',
          'early_stopping_rounds': 10,
          'eval_set': ['x_test', 'y_test']}

rs_clf = RandomizedSearchCV(clf, param_grid, n_iter=20,
                            n_jobs=1, verbose=2, cv=2,
                            scoring='neg_log_loss', refit=False, random_state=42)
print("Randomized search..")

rs_clf.fit(X_train, y_train)
best_score = rs_clf.best_score_
best_params = rs_clf.best_params_
print("Best score: {}".format(best_score))
print("Best params: ", best_params)

xgboostModel = XGBRFClassifier(subsample=.5, silent=False, reg_lambda=10, n_estimators=100, min_child_weight=.5,
                               max_depth=10, learning_rate=3, gamma=0, colsample_bytree=.8, colsample_bylevel=.4)
xgboostModel.fit(X_train, y_train)
predicted1 = xgboostModel.predict(X_test)

print('The score new XGBoost(train) is : ', xgboostModel.score(X_train, y_train))
print('The score new XGBoost(test) is : ', xgboostModel.score(X_test, y_test))

df_test = pd.DataFrame(X_test, columns=['PC1', 'PC2'])
df_test['Index'] = y_test
df_test['prediction'] = predicted1

sns.lmplot('PC1', 'PC2', hue='Index', data=df_test, fit_reg=False)
plt.title('new XGBoost (test--real)')
plt.show()
sns.lmplot('PC1', 'PC2', hue='prediction', data=df_test, fit_reg=False)
plt.title('new XGBoost (test--prediction)')
plt.show()

# accuracy score
print('-----------------classification_report of xgboost-------------------')
print(metrics.classification_report(y_test, predicted1))

sns.heatmap(metrics.confusion_matrix(y_test, predicted1), annot=True)
plt.title("Confusion Matrix 1")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

