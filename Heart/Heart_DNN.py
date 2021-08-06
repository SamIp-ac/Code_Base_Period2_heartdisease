import sys
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTE
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics

# Checking na
dfc = pd.read_csv(r'../heart.csv')
dfn = np.array(dfc)
features = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall'
            ]
print('The empty entries of df is : ', len(np.where(np.isnan(dfn))[0]))
print('The empty position of df is : ', np.where(np.isnan(dfn))[0])
# or using
dfc = pd.DataFrame(dfc)
nan_values = dfc[dfc.isna().any(axis=1)]
print(nan_values)
# import data
df = pd.read_csv(r'../heart.csv', skip_blank_lines=True)
df = pd.DataFrame(df)

df = df.dropna()

dataset = df[features]
result = df['output']
y = result.values
X = df[features].values

# numerical
count = Counter(y)
print('The number of 0,1 : ', count)

# data imbalance
balance = SMOTE()
X, y = balance.fit_resample(X, y)

counter = Counter(y)
print('The number of 0,1 : ', counter)
# Standardization, normalization

# scaler = StandardScaler()
# X = scaler.fit_transform(X)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, random_state=77)

# define the keras model
model = Sequential()
model.add(Dense(24, input_dim=dataset.shape[1], activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=150, batch_size=10)

# predict the model
predicted = model.predict_classes(X_train)
df_train = pd.DataFrame(X_train, columns=['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng',
                                          'oldpeak', 'slp', 'caa', 'thall'])
df_train['index'] = y_train
df_train['prediction'] = predicted

# calculate error
rmse = metrics.mean_squared_error(y_train, predicted, squared=False)
print('The root mean squared error of train is : ', rmse)
mae = metrics.mean_absolute_error(y_train, predicted)
print('The mean absolute error of train is : ', mae)
r2 = metrics.r2_score(y_train, predicted)
print('The r-squared of train is : ', r2)

# evaluate the keras model
_, accuracy = model.evaluate(X_train, y_train)
print('Accuracy of train data : ', (accuracy*100), ' %')

# test
predicted = model.predict_classes(X_test)
df_test = pd.DataFrame(X_test, columns=['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng',
                                        'oldpeak', 'slp', 'caa', 'thall'])
df_test['index'] = y_test
df_test['prediction'] = predicted

# calculate accuracy score
rmse = metrics.mean_squared_error(y_test, predicted, squared=False)
print('The root mean squared error of test is : ', rmse)
mae = metrics.mean_absolute_error(y_test, predicted)
print('The mean absolute error of test is : ', mae)
r2 = metrics.r2_score(y_test, predicted)
print('The r-squared of test is : ', r2)

# evaluate the keras model
n = 0
for i in range(0, np.array(df_test.shape[0])):
    if y_test[i] == predicted[i]:
        n = n + 1

print('The accuracy of test data : ', n/np.array(df_test.shape[0]))

_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy of test data : ', (accuracy*100), '%')

f1_score = metrics.f1_score(y_test, predicted)
print('The f1 score test data : ', f1_score)

print('-----------------classification_report of stacking-------------------')
print(metrics.classification_report(y_test, predicted))

sns.heatmap(metrics.confusion_matrix(y_test, predicted), annot=True)
plt.title("Confusion Matrix DNN")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
print(sys.version)
