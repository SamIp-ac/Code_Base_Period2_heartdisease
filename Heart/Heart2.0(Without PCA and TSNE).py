import random
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

# Dataset comes from Kaggle
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
# General overview
sns.displot(x='age', hue='sex', data=df, alpha=0.6)
plt.title('Gender')
plt.show()

plt.hist(result)
plt.show()

# Correlation
M = df
correlated = M.corr()
sns.heatmap(correlated, annot=True)
plt.show()

# numerical
counter = Counter(y)
print('The number of 0,1 : ', counter)

# data imbalance
over = SMOTE()
X, y = over.fit_resample(X, y)

# Standardize or Normalize
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=77)

# Croos_val_score
np.random.seed(77)
cv = cross_val_score(KNeighborsClassifier(), X, y, cv=5, scoring='accuracy')
score = np.mean(cv)
print('The cross-val score of Knn is : ', score)

np.random.seed(77)
cv = cross_val_score(RandomForestClassifier(), X, y, cv=5, scoring='accuracy')
score = np.mean(cv)
print('The cross-val score of RandomForest is : ', score)

np.random.seed(77)
cv = cross_val_score(DecisionTreeClassifier(), X, y, cv=5, scoring='accuracy')
score = np.mean(cv)
print('The cross-val score of DecisionTree is : ', score)

np.random.seed(77)
cv = cross_val_score(XGBRFClassifier(eval_metric='logloss', use_label_encoder=False), X, y, cv=5, scoring='accuracy')
score = np.mean(cv)
print('The cross-val score of xgboost is : ', score)

np.random.seed(77)
cv = cross_val_score(svm.SVC(), X, y, cv=5, scoring='accuracy')
score = np.mean(cv)
print('The cross-val score of SVM is : ', score)

np.random.seed(77)
cv = cross_val_score(GradientBoostingClassifier(), X, y, cv=5, scoring='accuracy')
score = np.mean(cv)
print('The cross-val score of GradientBoosting is : ', score)

np.random.seed(77)
cv = cross_val_score(GaussianNB(), X, y, cv=5, scoring='accuracy')
score = np.mean(cv)
print('The cross-val score of GaussianNB is : ', score)

np.random.seed(77)
cv = cross_val_score(SGDClassifier(), X, y, cv=5, scoring='accuracy')
score = np.mean(cv)
print('The cross-val score of SGDC is : ', score)

#####
np.random.seed(77)
estimators = [('knn', KNeighborsClassifier()),
              ('rf', RandomForestClassifier()),
              ('dt', DecisionTreeClassifier()),
              ('svm', svm.SVC()),
              ('xgboost', XGBRFClassifier(eval_metric='logloss', use_label_encoder=False))]
stackingCl = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
######
cv = cross_val_score(stackingCl, X, y, cv=5, scoring='accuracy')
score = np.mean(cv)
print('The cross-val score of stacking is : ', score)

# predict---xgboost
np.random.seed(77)
xgboostModel = XGBRFClassifier(n_estimators=100, learning_rate=0.3)
xgboostModel.fit(X_train, y_train)

predicted1 = xgboostModel.predict(X_test)

print('The score XGBoost(test) is : ', xgboostModel.score(X_test, y_test))

# features importance
xgboostModel.fit(dataset, result)
plot_importance(xgboostModel)
plt.show()

# accuracy score
print('-----------------classification_report of xgboost-------------------')
print(metrics.classification_report(y_test, predicted1))
print('jaccard_similarity_score', metrics.jaccard_score(y_test, predicted1))
print('log_loss', metrics.log_loss(y_test, predicted1))
print('zero_one_loss', metrics.zero_one_loss(y_test, predicted1))
print('AUC&ROC', metrics.roc_auc_score(y_test, predicted1))
print('matthews_corrcoef', metrics.matthews_corrcoef(y_test, predicted1))

# predict--stacking
np.random.seed(77)
estimators = [('knn', KNeighborsClassifier()),
              ('rf', RandomForestClassifier()),
              ('dt', DecisionTreeClassifier()),
              ('svm', svm.SVC()),
              ('xgboost', XGBRFClassifier(eval_metric='logloss', use_label_encoder=False))]
stackingCl = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stackingCl.fit(X_train, y_train)

predicted2 = stackingCl.predict(X_test)

print('The score stacking(test) is : ', stackingCl.score(X_test, y_test))

# accuracy score
print('-----------------classification_report of stacking-------------------')
print(metrics.classification_report(y_test, predicted2))
print('jaccard_similarity_score', metrics.jaccard_score(y_test, predicted2))
print('log_loss', metrics.log_loss(y_test, predicted2))
print('zero_one_loss', metrics.zero_one_loss(y_test, predicted2))
print('AUC&ROC', metrics.roc_auc_score(y_test, predicted2))
print('matthews_corrcoef', metrics.matthews_corrcoef(y_test, predicted2))

# Hyper-parameter tuning

tuned_parameters = [{'penalty': ['l1', 'l2'],
                     'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
                     'solver': ['liblinear'],
                     'multi_class': ['ovr']},
                    {'penalty': ['l2'],
                     'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
                     'solver': ['lbfgs'],
                     'multi_class': ['ovr', 'multinomial']}]

clf = GridSearchCV(LogisticRegression(tol=1e-6), tuned_parameters, cv=10)
clf.fit(X_train, y_train)
print('Best parameters set found for LR :', clf.best_params_)

# xgboost: {'solver': 'liblinear', 'penalty': 'l2', 'multi_class': 'ovr', 'C': 0.001}
# LR: {'solver': 'liblinear', 'penalty': 'l2', 'multi_class': 'ovr', 'C': 0.1}

model = LogisticRegression(penalty='l1', C=0.01, solver='liblinear', multi_class='ovr')
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
# LR
model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', multi_class='ovr')
model.fit(X_train, y_train)
model.score(X_test,y_test)
