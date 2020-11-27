# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

# %%
dataset = pd.read_csv("final_3.csv")

# %%
#Remove any invalid infinite value
dataset = dataset.replace([np.inf,-np.inf], np.nan)
dataset = dataset.dropna()

X = dataset
y = X[['Hindex_Growth_10years']]
X = dataset.drop(['Hindex_Growth_10years'], axis = 1)

print(X.head())

X1 = X
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, random_state = 1)

# %%
#import important packages
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_regression, mutual_info_regression

from sklearn.preprocessing import MinMaxScaler

# %%
scaler = MinMaxScaler()
scaled_X = scaler.fit_transform(X_train[X_train.columns[X_train.columns!= 'Hindex_Growth_10years' ]])
X_train = pd.DataFrame(scaled_X, columns = X_train.columns[X_train.columns!= 'Hindex_Growth_10years' ])
scaled_X_test = scaler.transform(X_test)
X_test = pd.DataFrame(scaled_X_test, columns = X_train.columns[X_train.columns!= 'Hindex_Growth_10years' ])

# %%
from sklearn.feature_selection import f_regression

# %%
select = SelectKBest(f_regression, k= 15)
selected_features = select.fit(X_train, y_train)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [X.columns[i] for i in indices_selected]

X_train_selected = X_train[colnames_selected]
X_test_selected = X_test[colnames_selected]
colnames_selected
selected_features.scores_
indices = np.argsort(selected_features.scores_)[::-1]

features = []
for i in range(15):
    features.append(X_train.columns[indices[i]])

# plot important variables
plt.figure()
plt.barh(features, selected_features.scores_[indices[range(15)]], color='b', align='center')
plt.show()

# %%
#plot heatmap to measure correlation
XY_train = pd.concat([X_train[features], y_train], axis = 1)

plt.figure(figsize=(15,15))
cor = (XY_train).corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# %%
X_train.head()

# %%
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score
reg = LinearRegression().fit(X_train_selected, y_train)
reg.score(X_train_selected, y_train)
print(reg.coef_)
print(reg.intercept_)
y_pred_reg = pd.DataFrame(reg.predict(X_test_selected))
rmse = np.sqrt(np.sum((reg.predict(X_test_selected)-y_test)**2)/len(y_test))
print(rmse)
r2_score(y_test, y_pred_reg)

# %%
from sklearn.linear_model import Ridge
reg1 = Ridge(alpha= 0.5)
reg1.fit(X_train_selected, y_train)
y_pred_reg1 = reg1.predict(X_test_selected)
reg1.score(X_train_selected, y_train)
print(reg1.coef_)
print(reg1.intercept_)
y_pred_reg1 = pd.DataFrame(reg1.predict(X_test_selected))
rmse = np.sqrt(np.sum((y_pred_reg1-y_test)**2)/len(y_test))
print(rmse)
r2_score(y_test, y_pred_reg1)

# %%
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=15, random_state=1)
regressor.fit(X_train_selected, y_train)
y_pred = regressor.predict(X_test_selected)
regressor.score(X_train_selected, y_train)

# %%
from sklearn.metrics import mean_squared_error 
from math import sqrt
rmse_val = [] #to store rmse values for different k
rsq = []
for K in range(10,50, 5):
    K = K+1
    regressor = RandomForestRegressor(n_estimators=K, random_state=0)
    regressor.fit(X_train_selected, y_train)
    y_pred = regressor.predict(X_test_selected)
    sq = regressor.score(X_train_selected, y_train)
    
    #model.fit(X_train_selected, y_train)  #fit the model
    #pred=regressor.predict(X_test_selected) #make prediction on test set
    error = sqrt(mean_squared_error(y_test,y_pred)) #calculate rmse
    
    rmse_val.append(error) #store rmse values
    rsq.append(sq)
    print('RMSE value for k= ' , K , 'is:', error)

#plotting the rmse values against k values


# %%
curve = pd.DataFrame(rmse_val) #elbow curve 
curve.plot()
plt.xlabel('Estimators')
plt.ylabel('RMSE')

# %%
curve = pd.DataFrame(rsq) #elbow curve 
curve.plot()
plt.xlabel('Estimators')
plt.ylabel('R-Square')

# %%
from sklearn import metrics
#y_pred = model.predict(X_test_selected)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# %%
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential

# %%
# Initialising the ANN
model = Sequential()

# Adding the input layer and the first hidden layer
model.add(Dense(32, activation = 'relu', input_dim = 15))

# Adding the second hidden layer
model.add(Dense(units = 32, activation = 'relu'))

# Adding the third hidden layer
model.add(Dense(units = 32, activation = 'relu'))

# Adding the output layer
model.add(Dense(units = 1))

# %%
model.compile(optimizer = 'adam',loss = 'mean_squared_error')

# %%
model.fit(X_train_selected, y_train, batch_size = 10, epochs = 10)

# %%
from sklearn import metrics
y_pred = model.predict(X_test_selected)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# %%
#print(r2_score(y_test, y_pred))
r2_score(y_test, y_pred)

# %%
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
import matplotlib.pyplot as plt
#matplotlib inline
rmse_val = [] #to store rmse values for different k
for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train_selected, y_train)  #fit the model
    pred=model.predict(X_test_selected) #make prediction on test set
    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)

#plotting the rmse values against k values
curve = pd.DataFrame(rmse_val) #elbow curve 
curve.plot()


reg3 = neighbors.KNeighborsRegressor(n_neighbors = 12)
reg3.fit(X_train_selected, y_train)
y_pred_reg3 = reg3.predict(X_test_selected)
reg3.score(X_train_selected, y_train)

y_pred_reg3 = pd.DataFrame(reg3.predict(X_test_selected))
rmse = np.sqrt(np.sum((y_pred_reg3-y_test)**2)/len(y_test))
rmse
print(rmse)
print(r2_score(y_test, y_pred_reg3))
r2_score(y_test, y_pred_reg3)
both3 = y_test.reset_index()
both3['Pred'] = y_pred_reg3

# %%
