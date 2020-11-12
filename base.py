import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer

data = pd.read_csv("citiesfinal.csv", sep=";")
variables = data.columns

# number of na by column
nascol=data.isna().sum(axis=0)

# correlation with CO2 emissions
emissioncorr = data.corr()['CO2 Emissions per Capita (metric tonnes)']

# highly correlated variables
hcorr = emissioncorr.index[abs(emissioncorr>0.3)]

# keeping variables with less than 30 na or highly correlated
df = data[variables[nascol<30] | hcorr]

# impute the highly correlated variables
dfimp= data[hcorr]  #variables[nascol>30] &
# from Gulin's code:
imputer = KNNImputer(n_neighbors=2, weights="uniform")
# The Imputer gives an array as an output
imputed_array = imputer.fit_transform(dfimp)
# Putting the array into a dataframe
dfimp = pd.DataFrame(imputed_array, columns=dfimp.columns)

# Insert back the columns
df[hcorr]=dfimp #variables[nascol>30] &

# drop rows that still have missing values
df = df.dropna(axis=0)

# Correlation
correlatedvar = abs(df.corr()[['CO2 Emissions per Capita (metric tonnes)']]).sort_values(by='CO2 Emissions per Capita (metric tonnes)', ascending=False).index.to_list()


#Random forest regressor optimization for variables and in number of estimators
list=[]
listr2=[]
for i in np.arange(5,20): #number of variables
    for n in np.arange(5,20): #number of forest estimators
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
        listr2.append(R2)
opt=list[listr2.index(max(listr2))]
opt

