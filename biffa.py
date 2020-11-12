import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

data = pd.read_csv("citiesfinal.csv", sep=",")

variables = data.columns


# cutting nas
# by column
nascol=data.isna().sum(axis=0)
df=data[variables[nascol<30]]

# by row
df=df.dropna(axis=0)

# correlation
correlatedvar=abs(df.corr()[['CO2 Emissions per Capita (metric tonnes)']]).sort_values(by='CO2 Emissions per Capita (metric tonnes)', ascending=False).index.to_list()


#outlier dropping larger than
emissions=data['CO2 Emissions per Capita (metric tonnes)']
data=data[data['CO2 Emissions per Capita (metric tonnes)']>np.var(emissions),:]


###################################################
plt.figure()
sb.distplot(data['CO2 Emissions per Capita (metric tonnes)'],bins=50)
plt.figure()
plt.plot(nascol)
plt.plot(abs(data.corr()[['CO2 Emissions per Capita (metric tonnes)']]*150))
nascol.plot()
abs(data.corr()[['CO2 Emissions per Capita (metric tonnes)']]*150).plot()
# insight visualization
emissions = data['CO2 Emissions per Capita (metric tonnes)']
# m
dataf = data.loc[:, 'Car Modeshare (%)':].fillna(data.loc[:, 'Car Modeshare (%)':].mean(), inplace=True)
for col in data.columns[9]: data[col].fillna(value=data[col].mean(), inplace=True)

scatter=data[correlatedvar[:4]]
plt.figure()
g=sb.PairGrid(scatter, diag_sharey=False)
g.map_upper(sb.scatterplot, s=15)
g.map_lower(sb.kdeplot)
g.map_diag(sb.kdeplot, lw=2)
plt.figure()
emissions.hist()




#Models
#Train and test

#linear regression
list=[]
for i in np.arange(5,20):
    dfcorr=df[correlatedvar[:i]]
    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler(feature_range=(1,10))
    dfscal=scaler.fit_transform(dfcorr)
    Y=dfscal[:,0]
    X=dfscal[:,1:]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,shuffle=False)
    regr=linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    prediction=regr.predict(X_test)
    R2=sklearn.metrics.r2_score(y_test,prediction)
    list.append(R2)
print('optimal amount of variables: {}, R2=' .format(list.index(max(list))+5),R2) #max in 12


#Lasso
dfcorr=df[correlatedvar]
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(1,10))
dfscal=scaler.fit_transform(dfcorr)
Y=dfscal[:,0]
X=dfscal[:,1:]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,shuffle=False)
reg = linear_model.Lasso(alpha=0.1)
reg.fit(X_train, y_train)
prediction=reg.predict(X_test)
R2=sklearn.metrics.r2_score(y_test,prediction)
print('R2 ={}'.format(R2))


#generalized not orking
from sklearn.linear_model import TweedieRegressor
list=[]
for i in np.arange(5,20):
    dfcorr=df[correlatedvar[:i]]
    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler(feature_range=(1,10))
    dfscal=scaler.fit_transform(dfcorr)
    Y=dfscal[:,0]
    X=dfscal[:,1:]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,shuffle=False)
    regr=TweedieRegressor(power=1, alpha=0.5, link='log')
    regr.fit(X_train, y_train)
    prediction=regr.predict(X_test)
    R2=sklearn.metrics.r2_score(y_test,prediction)
    list.append(R2)
print('optimal amount of variables: {}, R2=' .format(list.index(max(list))+5),R2) #max in 12


#polynomial
from sklearn.preprocessing import PolynomialFeatures
list=[]
for i in np.arange(2,10):
    dfcorr=df[correlatedvar[:i]]
    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler(feature_range=(1,10))
    dfscal=scaler.fit_transform(dfcorr)
    Y=dfscal[:,0]
    X=dfscal[:,1:]
    poly = PolynomialFeatures(degree=3)
    poly.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,shuffle=False)
    regr=linear_model.LinearRegression(fit_intercept=False)
    regr.fit(X_train, y_train)
    prediction=regr.predict(X_test)
    R2=sklearn.metrics.r2_score(y_test,prediction)
    list.append(R2)
print('optimal amount of variables: {}, R2=' .format(list.index(max(list))+5),R2) #max in 12




#ann
list=[]
from sklearn.neural_network import MLPRegressor
for i in np.arange(5,20):
    dfcorr = df[correlatedvar[:i]]
    scaler = MinMaxScaler(feature_range=(1, 10))
    dfscal = scaler.fit_transform(dfcorr)
    Y = dfscal[:, 0]
    X = dfscal[:, 1:]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,shuffle=False)
    regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
    pred=regr.predict(X_test)
    R2=sklearn.metrics.r2_score(y_test,pred)
    list.append(R2)
    list
print('optimal amount of variables: {}, R2 =' .format(list.index(max(list))+5),R2) #max in 10 variables most of the times


#voting regressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
list=[]
for i in np.arange(5,19):
    dfcorr = df[correlatedvar[:i]]
    scaler = MinMaxScaler(feature_range=(1, 10))
    dfscal = scaler.fit_transform(dfcorr)
    Y = dfscal[:, 0]
    X = dfscal[:, 1:]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,shuffle=False)
    reg1 = GradientBoostingRegressor(random_state=1, n_estimators=10)
    reg2 = RandomForestRegressor(random_state=1, n_estimators=10)
    reg3 = LinearRegression()
    ereg = VotingRegressor(estimators=[('gb', reg2), ('rf', reg3),('lr', reg3)])
    ereg = ereg.fit(X_train, y_train)
    pred=ereg.predict(X_test)
    R2=sklearn.metrics.r2_score(y_test,pred)
    list.append(R2)
print('optimal amount of variables: {}, R2 =' .format(list.index(max(list))+5),R2) #max in 34 R2=0.6775



list=[]
listr=[]
for i in np.arange(5,20):
    for n in np.arange(5,20):
        dfcorr = df[correlatedvar[:i]]
        scaler = MinMaxScaler(feature_range=(1, 10))
        dfscal = scaler.fit_transform(dfcorr)
        Y = dfscal[:, 0]
        X = dfscal[:, 1:]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,shuffle=False)
        reg = RandomForestRegressor(random_state=1, n_estimators=n)
        reg = reg.fit(X_train, y_train)
        pred=reg.predict(X_test)
        R2=sklearn.metrics.r2_score(y_test,pred)
        list.append([i,n,R2])
        listr.append(R2)
max=list[listr.index(max(listr))]
list[max]
print('optimal var={}, estim={} R2 ={}' .format(max[1],max[2],max[3])) #max in 34 R2=0.6775
#0.7199712887400289 12 corrvar 14 n estim
list




from sklearn.cluster import KMeans
dfcorr = df[correlatedvar]
scaler = MinMaxScaler(feature_range=(1, 10))
dfscal = scaler.fit_transform(dfcorr)
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(dfscal)
scatter=df[correlatedvar[:4]]
scatter['clust']=kmeans.labels_
plt.figure()
g=sb.PairGrid(scatter, hue='clust')
g.map_upper(sb.scatterplot, s=15)
g.map_lower(sb.kdeplot)
g.map_diag(sb.kdeplot, lw=2)

sb.histplot(x=scatter['CO2 Emissions per Capita (metric tonnes)'],hue=scatter['clust'])
plt.figure()
sb.distplot(scatter['CO2 Emissions per Capita (metric tonnes)'][scatter.clust==1])
sb.distplot(scatter['CO2 Emissions per Capita (metric tonnes)'][scatter.clust==0])






X=data[correlatedvar].dropna()
y=data['CO2 Emissions per Capita (metric tonnes)'].iloc[X.index]

sc = StandardScaler()
X = sc.fit_transform(X)
y = sc.fit_transform(y.values.reshape(-1,1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


regr=linear_model.LinearRegression(fit_intercept=False)
regr.fit(X_train, y_train)
pred=regr.predict(X_test)
R2=sklearn.metrics.r2_score(y_test,pred)




# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=19, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the training set
train=model.fit(X_train, y_train, epochs=100, batch_size=10)


# make probability predictions with the model
predictions = model.predict(X_test)

predictions

R2=sklearn.metrics.r2_score(y_test,predictions)


plt.figure()
plt.plot(predictions,label="pred ANN")
plt.plot(predictions,label="pred regr")
plt.plot(y_test,label="test")
plt.title("%f" %R2)
plt.legend()




#####################
####   TRASH   ######
#train and test



correlatedvar=data.corr()[['CO2 Emissions per Capita (metric tonnes)']].sort_values(by='CO2 Emissions per Capita (metric tonnes)', ascending=False)[1:11].index.to_list()

X=data[correlatedvar]
y=data['CO2 Emissions per Capita (metric tonnes)']

sklearn



variables = data.columns




#target variable
emissions = data['CO2 Emissions per Capita (metric tonnes)']

#sum of nans
emissions.corr()



corr = data.corr()
sb.heatmap(corr)


sb.heatmap(data.corr()[['CO2 Emissions per Capita (metric tonnes)']])




CO2corr=corr['CO2 Emissions per Capita (metric tonnes)']

nas=data


data.dropna

emissioncorr=corr['CO2 Emissions per Capita (metric tonnes)']

nascol=data.isna().sum(axis=0)
nasrow=data.isna().sum(axis=1)

nas.plot()
emissioncorr.plot()

plt.hist(nasrow)


plt.figure()
fig,ax1=plt.subplots()
ax2=ax1.twinx()
ax1.plot(emissioncorr)
ax2.plot(nascol)





datafilt=data[variables[nascol<30]]

datafilt2=datafilt.dropna(axis=0)




# Variabili correlate

correlatedvar=data.corr()[['CO2 Emissions per Capita (metric tonnes)']].sort_values(by='CO2 Emissions per Capita (metric tonnes)', ascending=False)[1:10].index.to_list()



X=datafilt2[correlatedvar]


#culin

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 12:36:33 2020

@author: gulin
"""
#To run current cell - kntrl+enter
#To run selected line or selection F9

#%% Importing packages
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.impute import KNNImputer

pd.set_option('display.max_rows', None)

#%%
#Reading the data
df = pd.read_csv("citiesfinal.csv", sep=',')

#Deleting the unnecessary column
del df["Unnamed: 0"]

#Setting the index as city name
df = df.set_index("City")

#A list of all the column names
col_names = df.columns  # variables (columns) in the dataset

#%%Looking at NAs
#The number of missing values in each column
nascol = df.isna().sum(axis=0)  # number of nas per column/variable
nasrow = df.isna().sum(axis=1)

#When you drop rows with any NA the shape is (23,77)
df.dropna(how='any', axis=0).shape

#When you drop columns with any NA the shape is (331,29)
df.dropna(how='any', axis=1).shape

#Counting the number of NaN cells in the entire data set (There are 3201 empty cells)
x= df.isnull().sum().sum()

#Percentage of NaNs in the dataset (Excluding the clusters and the factors) (It's 15.2%)
perc_na = x / (330*(77-4-9))

#%%Data Imputation
df2 = df.drop(['cityID', 'clusterID','Typology','Country'], axis = 1)
df3= df2.drop(col_names[68:77],axis=1)
col_names_array = df3.columns

#Using KNN Imputation
imputer = KNNImputer(n_neighbors=2, weights="uniform")
#The Imputer gives an array as an output
imputed_array = imputer.fit_transform(df3)

#Putting the array into a dataframe
dfimp = pd.DataFrame(imputed_array, columns=col_names_array)

col_list=["City"]
#Look at the descriptive statistics (also for the initial case)
cities = pd.read_csv("citiesfinal.csv", sep=',',usecols=col_list)
y=df.index
dfimp.set_index(cities.City)

#To get the city names again in the new dataframe
dfimp = pd.concat([dfimp, cities], axis=1, sort=False)
dfimp = dfimp.set_index("City")

#Looking at the descriptive statistics before and after imputation
descriptive_new=dfimp.describe()
descriptive_bef=df.describe()
descriptive_bef=descriptive_bef.drop(col_names[68:77],axis=1)
descriptive_bef=descriptive_bef.drop(['cityID', 'clusterID'],axis=1)
#To see how the mean changes after KNN Imptuation
change = (descriptive_new - descriptive_bef)/descriptive_bef

#%%PCA Trial
#Need to rescale every variable to prepare for PCA


#%%Model trial
Y = dfimp["CO2 Emissions per Capita (metric tonnes)"]
X = dfimp.drop(["CO2 Emissions per Capita (metric tonnes)"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, shuffle=False)
reg = RandomForestRegressor(random_state=1, n_estimators=n)
reg = reg.fit(X_train, y_train)
pred = reg.predict(X_test)
R2 = sklearn.metrics.r2_score(y_test, pred)
