from dateutil.parser import parse
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
#plt.rcParams.update({'figure.figsize': (10, 7), 'figure.dpi': 120})
#df = pd.read_csv('air_passengers.csv', parse_dates = ['Month'], index_col = ['Month'])

df=pd.read_csv('C:/Users/45526/Desktop/DTU/Thesis/Data_cons.csv',skiprows=2,index_col=['READ_TIME_Date'])
df = df.rename(columns = {'priser euro': 'Prices'}, inplace = False)
dfWeekDays=df[df.weekend!=1]
dfWeekend=df[df.weekend==1]
dfWeekDays = dfWeekDays.rename(columns = {'priser euro': 'Prices'}, inplace = False)
#g=sns.lineplot(dfWeekDays.Hour,dfWeekDays.Prices)
#g=sns.lineplot(dfWeekend.Hour,dfWeekend.Prices)

new_labels = ['weekdays prices', 'weekend prices']

#Aggregated considering different typology of house
WDaverage=dfWeekDays.loc[:, dfWeekDays.columns.isin(k for k in dfWeekDays.columns if 'Average' in k)]
WDsum=dfWeekDays.loc[:, dfWeekDays.columns.isin(k for k in dfWeekDays.columns if 'Sum' in k)]
WDAntal=dfWeekDays.loc[:, dfWeekDays.columns.isin(k for k in dfWeekDays.columns if 'Antal' in k)]


#4 classes (sum,average and number of smart meters)
#Consumers With EV but without Eh
WDAntalEVnoEh=WDAntal.loc[:, WDAntal.columns.isin(k for k in WDAntal if 'mEV21' in k or 'mEV11' in k)]
NumEVnoEh=WDAntalEVnoEh.sum(axis=1)
WDaverageEVnoEh=WDaverage.loc[:, WDaverage.columns.isin(k for k in WDaverage if 'mEV21' in k or 'mEV11' in k)]
MeanEVnoEh=WDaverageEVnoEh.mean(axis=1)
WDEVnoEhsum=WDsum.loc[:, WDsum.columns.isin(k for k in WDsum if 'mEV21' in k or 'mEV11' in k)]
SumEVnoEh=WDEVnoEhsum.sum(axis=1)

#Consumers with EV and Eh
WDAntalEVEh=WDAntal.loc[:, WDAntal.columns.isin(k for k in WDAntal if 'mEV22' in k or 'mEV12' in k)]
NumEVEh=WDAntalEVEh.sum(axis=1)
WDaverageEVEh=WDaverage.loc[:, WDaverage.columns.isin(k for k in WDaverage if 'mEV22' in k or 'mEV12' in k)]
MeanEVEh=WDaverageEVEh.mean(axis=1)
WDEVEhsum=WDsum.loc[:, WDsum.columns.isin(k for k in WDsum if 'mEV22' in k or 'mEV12' in k)]
SumEVEh=WDEVEhsum.sum(axis=1)

#Consumers without EV with Eh
WDAntalnoEVEh=WDAntal.loc[:, WDAntal.columns.isin(k for k in WDAntal if 'uEV22' in k or 'uEV12' in k)]
NumnoEVEh=WDAntalnoEVEh.sum(axis=1)
WDaveragenoEVEh=WDaverage.loc[:, WDaverage.columns.isin(k for k in WDaverage if 'uEV22' in k or 'uEV12' in k)]
MeannoEVEh=WDaveragenoEVEh.mean(axis=1)
WDnoEVEhsum=WDsum.loc[:, WDsum.columns.isin(k for k in WDsum if 'uEV22' in k or 'uEV12' in k)]
SumnoEVEh=WDnoEVEhsum.sum(axis=1)

#Consumers without EV without Eh
WDAntalnoEVnoEh=WDAntal.loc[:, WDAntal.columns.isin(k for k in WDAntal if 'uEV21' in k or 'uEV11' in k)]
NumnoEVnoEh=WDAntalnoEVnoEh.sum(axis=1)
WDaveragenoEVnoEh=WDaverage.loc[:, WDaverage.columns.isin(k for k in WDaverage if 'uEV21' in k or 'uEV11' in k)]
MeannoEVnoEh=WDaveragenoEVnoEh.mean(axis=1)
WDnoEVnoEhsum=WDsum.loc[:, WDsum.columns.isin(k for k in WDsum if 'uEV21' in k or 'uEV11' in k)]
SumnoEVnoEh=WDnoEVnoEhsum.sum(axis=1)

#New dataframe with the for classes with sum average and total number
total4Classes=pd.concat((df.Hour,df.month,NumEVEh,SumEVEh,MeanEVEh,NumnoEVEh,SumnoEVEh,MeannoEVEh,NumEVnoEh,SumEVnoEh,MeanEVnoEh,NumnoEVnoEh,SumnoEVnoEh,MeannoEVnoEh),axis=1)
total4Classes.columns=['Hour','month','NumEVEh','SumEVEh','MeanEVEh','NumnoEVEh','SumnoEVEh','MeannoEVEh','NumEVnoEh','SumEVnoEh','MeanEVnoEh','NumnoEVnoEh','SumnoEVnoEh','MeannoEVnoEh']

Mean4classes=pd.concat((df.Hour,df.month,MeanEVEh,MeanEVnoEh,MeannoEVEh,MeannoEVnoEh),axis=1).dropna()
Mean4classes.columns=['Hour','month','EVEh','EVnoEh','noEVEh','noEVnoEh']


#plot of differences of mean combination with/without EV Eh
sns.lineplot(df.Hour,MeanEVEh,label='With EV With Eh')
sns.lineplot(df.Hour,MeanEVnoEh,label='With EV no Eh')
sns.lineplot(df.Hour,MeannoEVnoEh,label='no EV no Eh')
sns.lineplot(df.Hour,MeannoEVEh,label='no EV With Eh')


#seasonal Eh
Mean4classesWinter=Mean4classes[Mean4classes.month==(12 or 1 or 2 or 3)]
Mean4classesSummer=Mean4classes[Mean4classes.month==(6 or 7 or 8 )]
plt.figure()
sns.lineplot(Mean4classesWinter.Hour,Mean4classesWinter.EVEh,label='With EV With Eh Winter')
sns.lineplot(Mean4classesWinter.Hour,Mean4classesWinter.EVnoEh,label='With EV no Eh Winter')
sns.lineplot(Mean4classesWinter.Hour,Mean4classesWinter.noEVnoEh,label='no EV no Eh Winter')
sns.lineplot(Mean4classesWinter.Hour,Mean4classesWinter.noEVEh,label='no EV With Eh Winter')
sns.lineplot(Mean4classesSummer.Hour,Mean4classesSummer.EVEh,label='With EV With Eh Summer')
sns.lineplot(Mean4classesSummer.Hour,Mean4classesSummer.EVnoEh,label='With EV no Eh Summer')
sns.lineplot(Mean4classesSummer.Hour,Mean4classesSummer.noEVnoEh,label='no EV no Eh Summer')
sns.lineplot(Mean4classesSummer.Hour,Mean4classesSummer.noEVEh,label='no EV With Eh Summer')


plt.figure()
#Electric heating increasing with winter months
months=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
for m in months:
    sns.lineplot(Mean4classes.Hour[Mean4classes.month==months.index(m)+1],Mean4classes.noEVEh,label='no EV With Eh month %s'%m,ci=0)















#Trash-------------------------------------------------------------
WinterAv=
SummerAv=


WDAvEv=WDaverage.loc[:,WDaverage.columns.isin(k for k in WDaverage.columns if 'mEV' in k)]

df.columns = [df.iloc[0,:]]
df=df.iloc[1:,:]
x=df.iloc[0:,].values
y=df.iloc[0:,176].values

print(df)
Prices=df[['priser euro']]
print(Prices)
#dti = df.to_datetime(['1/1/2018', np.datetime64('2018-01-01'),:datetime.datetime(2018, 1, 1)])
df.head()
def plot_df(df, x, y, title="", xlabel='Date', ylabel='Value', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

