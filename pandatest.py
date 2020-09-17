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



#Average consumption with or without EV
WDaverageEVnoEh=WDaverage.loc[:, WDaverage.columns.isin(k for k in WDaverage if 'mEV' in k and ('11' or '21') in k)]
WDaverageEVEh=WDaverage.loc[:, WDaverage.columns.isin(k for k in WDaverage if 'mEV' in k and ('12' or '22') in k)]


#plot of differences of mean combination with/without EV Eh
WDaveragemEV22=WDaverage.loc[:, WDaverage.columns.isin(k for k in WDaverage if 'mEV22' in k or 'mEV12' in k)]
EhEVMean=WDaveragemEV22.mean(axis=1)
WDaverageEVnoEh=WDaverage.loc[:, WDaverage.columns.isin(k for k in WDaverage if 'mEV11' in k or 'mEV21' in k)]
noEhEVMean=WDaverageEVnoEh.mean(axis=1)
WDaveragenoEVnoEh=WDaverage.loc[:, WDaverage.columns.isin(k for k in WDaverage if 'uEV11' in k or 'uEV21' in k)]
noEhnoEVMean=WDaveragenoEVnoEh.mean(axis=1)
WDaveragenoEVEh=WDaverage.loc[:, WDaverage.columns.isin(k for k in WDaverage if 'uEV22' in k or 'uEV12' in k)]
EhnoEVMean=WDaveragenoEVEh.mean(axis=1)

sns.lineplot(df.Hour,EhEVMean,label='With EV With Eh')
sns.lineplot(df.Hour,noEhEVMean,label='With EV no Eh')
sns.lineplot(df.Hour,noEhnoEVMean,label='no EV no Eh')
sns.lineplot(df.Hour,EhnoEVMean,label='no EV With Eh')


totalEVEh=pd.concat((df.Hour,df.month,EhEVMean,noEhnoEVMean,noEhEVMean,EhnoEVMean),axis=1).dropna()
totalEVEh.columns=['Hour','month','EhEV','noEhnoEV','noEhEV','EhnoEV']

#seasonal Eh
totalEVEhWinter=totalEVEh[totalEVEh.month==(12 or 1 or 2 or 3)]
totalEVEhSummer=totalEVEh[totalEVEh.month==(6 or 7 or 8 )]




sns.lineplot(df.Hour,EhEVMean,label='With EV With Eh')
sns.lineplot(df.Hour,noEhEVMean,label='With EV no Eh')
sns.lineplot(df.Hour,noEhnoEVMean,label='no EV no Eh')
sns.lineplot(df.Hour,EhnoEVMean,label='no EV With Eh')







sns.lineplot(totalEVEhWinter.Hour,totalEVEhWinter.EhEV,label='With EV With Eh Winter')
sns.lineplot(totalEVEhWinter.Hour,totalEVEhWinter.noEhEV,label='With EV no Eh Winter')
sns.lineplot(totalEVEhWinter.Hour,totalEVEhWinter.noEhnoEV,label='no EV no Eh Winter')
sns.lineplot(totalEVEhWinter.Hour,totalEVEhWinter.EhnoEV,label='no EV With Eh Winter')
sns.lineplot(totalEVEhSummer.Hour,totalEVEhSummer.EhEV,label='With EV With Eh Summer')
sns.lineplot(totalEVEhSummer.Hour,totalEVEhSummer.noEhEV,label='With EV no Eh Summer')
sns.lineplot(totalEVEhSummer.Hour,totalEVEhSummer.noEhnoEV,label='no EV no Eh Summer')
sns.lineplot(totalEVEhSummer.Hour,totalEVEhSummer.EhnoEV,label='no EV With Eh Summer')

#Electric heating increasing with winter months
months=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
for m in months:
    sns.lineplot(totalEVEh.Hour[totalEVEh.month==months.index(m)+1],totalEVEh.EhnoEV,label='no EV With Eh month %s'%m,ci=0)


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

