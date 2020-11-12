import pandas as pd
import seaborn as sb
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import pmdarima
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
import sklearn.metrics as metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates




df=pd.read_csv('Data kategorier.csv',skiprows=2,index_col=['READ_TIME_Date'])
df.index = pd.to_datetime(df.index)


#MAIN
size=(7,5)
plt.style.use('seaborn')
sb.set_style("darkgrid")
color_list = ['blue','red','green']


####### Class aggregation #########
def aggregate(data,conditions,name,weekend=True):
    #if not weekend:
    #    data=data[data.weekend != 1]
    if len(conditions)==2:
		Antal = data.loc[:,df.columns.isin(k for k in df.columns if 'Antal' in k and ( conditions[0] in k or conditions[1] in k))]
		Average = data.loc[:, df.columns.isin(k for k in df.columns if 'Average' in k and (conditions[0] in k or conditions[1] in k))]
		Tot = data.loc[:, df.columns.isin(k for k in df.columns if 'Sum' in k and (conditions[0] in k or conditions[1] in k))]
	elif len(conditions) == 3:
		Antal = data.loc[:,df.columns.isin(k for k in df.columns if 'Antal' in k and (conditions[0] in k or conditions[1] in k) and (conditions[2] in k))]
		Average = data.loc[:, df.columns.isin(k for k in df.columns if 'Average' in k and (conditions[0] in k or conditions[1] in k) and (conditions[2] in k))]
		Tot = data.loc[:,df.columns.isin(k for k in df.columns if 'Sum' in k and (conditions[0] in k or conditions[1] in k) and (conditions[2] in k))]
    Ant=Antal.mean(axis=1)
    Aver=Average.mean(axis=1)
    Sum=Tot.mean(axis=1)
    aggreg=pd.concat([Ant,Aver,Sum],axis=1)
    aggreg['type'] = name
	aggreg.columns=['Antal'+name,'Mean'+name,'Sum'+name,'type']
	return(aggreg)


#4classes based on electric heating and EV
EVEh=aggregate(df,['mEV22','mEV12'],'EVEh')
noEVnoEh=aggregate(df,['uEV21','uEV11'],'noEVnoEh')
EVnoEh=aggregate(df,['mEV21','mEV11'],'EVnoEh')
noEVEh=aggregate(df,['uEV22','uEV12'],'noEVEh')

dfEl=pd.concat([df.iloc[:,:4],EVEh,EVnoEh,noEVEh,noEVnoEh],axis=1)

EVEh.columns=['Antal','Mean','Sum','type']
noEVnoEh.columns=['Antal','Mean','Sum','type']
EVnoEh.columns=['Antal','Mean','Sum','type']
noEVEh.columns=['Antal','Mean','Sum','type']
dfp=pd.concat([EVEh,EVnoEh,noEVEh,noEVnoEh],axis=0)
dfp['Hour']=dfp.index.hour
#dfpivot = dfp.pivot(columns='Hour')


#classes population
sum(EVEh.Antal<30)
EVEh.Antal.describe()


#EV and EH winter and summer

plt.figure(figsize=size)
ax=sb.lineplot(dfp.index.hour,dfp.Mean,hue=dfp.type)
ax.set(xlabel='Hour', ylabel='Average household consumption [kW]')


AntalEV=df.loc[:,df.columns.isin(k for k in df.columns if 'Antal' in k and 'mEV' in k)]
AntalEV.max()

####EV vs NoEV

EV=pd.DataFrame({'Mean':(EVEh.Mean+EVnoEh.Mean)/2,'EV':'Yes'})
noEV=pd.DataFrame({'Mean':(noEVEh.Mean+noEVnoEh.Mean)/2,'EV':'no'})
EV=pd.DataFrame({'Mean':df.Average_h2vmba2mEV21,'EV':'Yes'})
noEV=pd.DataFrame({'Mean':(df.Average_h2vmba2uEV21)/2,'EV':'no'})
dfEV=pd.concat([EV,noEV],axis=0)
dfEV['dayn']=dfEV.index.weekday
dfEV['Day']=dfEV.dayn.apply(lambda x: 'weekend' if x>=5 else 'weekday')

plt.figure(figsize=size)
ax=sb.lineplot(dfEV.index.hour,dfEV.Mean,hue=dfEV.EV,style=dfEV.Day)
ax.set(xlabel='Hour', ylabel='Average household consumption [kW]')


#Seasonal
dfEV['Month']=dfEV.index.month_name(locale='Spanish')
dfEV['Monthn']=dfEV.index.month
dfEV['Season']=dfEV.Monthn.apply(lambda x: 'Winter' if x in [12,1,2] else('Spring' if x in [3,4,5] else('Summer' if x in [6,7,8] else ('Fall'))))

plt.figure(figsize=size)
ax=sb.lineplot(dfEV.index.hour,dfEV.Mean,hue=dfEV.EV,style=dfEV.Season)
ax.set(xlabel='Hour', ylabel='Average household consumption [kW]')


#difference in EV and Not EV

diff=pd.DataFrame({'Mean': EV.Mean-noEV.Mean})
diff['dayn']=diff.index.weekday
diff['Day']=diff.dayn.apply(lambda x: 'weekend' if x>=5 else 'weekday')
plt.figure(figsize=size)
ax=sb.lineplot(diff.index.hour,diff.Mean,style=diff.Day)
ax.set(xlabel='Hour', ylabel='Average difference EV-noEV [kW]')




#MAIN
size=(7,5)
plt.style.use('seaborn')
sb.set_style("darkgrid")
color_list = ['blue','red','green']


clustering = pd.DataFrame(index=months)
clustering["Features"]=[3,2,2,2,2,2,2,2,2,2,2,3]
clustering["Original"]=[2,2,2,3,2,2,3,3,2,2,2,2]

clustering.plot()
plt.figure()
plt.scatter(clustering.index,clustering.Features,label="Original")
plt.scatter(clustering.index,clustering.Original,label="Features")
plt.legend()