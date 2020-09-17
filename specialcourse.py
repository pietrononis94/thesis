import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import itertools as itt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import pmdarima



data=pd.read_csv("C:/Users/pietr/PycharmProjects/untitled1/Data kategorier.csv",skiprows=2, index_col=['READ_TIME_Date'], parse_dates=True)
data.index=pd.to_datetime(data.index)


#Cutting weekends
#data=data[data.weekend!=1]

#Chosing only average consumption
average=data.loc[:, data.columns.isin(k for k in data.columns if 'Average' in k)]
Antal=data.loc[:, data.columns.isin(k for k in data.columns if 'Average' in k)]
Sum=data.loc[:, data.columns.isin(k for k in data.columns if 'Average' in k)]


#Average consumption of aggregated categories: a=age, Eh=electric heating
a1Eh=data.loc[:, data.columns.isin(k for k in average.columns if '11' in k or '21' in k or 'a1' in k)].mean(axis=1)
a2Eh=data.loc[:, data.columns.isin(k for k in average.columns if '11' in k or '21' in k or 'a2' in k)].mean(axis=1)
a1noEh=data.loc[:, data.columns.isin(k for k in average.columns if '12' in k or '22' in k or 'a1' in k)].mean(axis=1)
a2noEh=data.loc[:, data.columns.isin(k for k in average.columns if '12' in k or '22' in k or 'a2' in k)].mean(axis=1)

mat=pd.concat((data.Hour,data.month,a1Eh,a2Eh,a1noEh,a2noEh),axis=1).dropna()
mat.columns=['Hour','month','a1Eh','a2Eh','a1noEh','a2noEh']


#whole year and average
sb.lineplot(mat.index,mat.a1Eh,size=1.5, data=mat.dropna())
rolling_mean = mat.rolling(window = 336).mean()
sb.lineplot(mat.time,rolling_mean.a1Eh, color="coral", label="14 days average trendline")
rolling_std = mat.a1Eh.rolling(window = 12).std()

#average whole year
sb.lineplot(mat.Hour,mat.a1Eh)
sb.lineplot(mat.Hour,mat.a2Eh)
sb.lineplot(mat.Hour,mat.a1noEh)
sb.lineplot(mat.Hour,mat.a2noEh)

#average weekday in summer and in winter
summer=mat[mat.month.isin(k for k in (6,7,8))]
summer=summer.iloc[:,[1,3,4,5,6]]
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (24, 6),sharey='row')
ax1.set_title("Summer hourly average")
sb.lineplot(summer.Hour,summer.a1Eh,label='Young el.heating',ax=ax1)
sb.lineplot(summer.Hour,summer.a2Eh,label='Old el.heating',ax=ax1)
sb.lineplot(summer.Hour,summer.a1noEh,label='Young no el.heating',ax=ax1)
sb.lineplot(summer.Hour,summer.a2noEh,label='Old no el.heating',ax=ax1)
winter=mat[mat.month.isin(k for k in (1,2,11,12))]
winter=winter.iloc[:,[1,3,4,5,6]]
ax2.set_title("Winter hourly average")
sb.lineplot(winter.Hour,winter.a1Eh,label='Young el.heating',ax=ax2)
sb.lineplot(winter.Hour,winter.a2Eh,label='Old el.heating',ax=ax2)
sb.lineplot(winter.Hour,winter.a1noEh,label='Young no el.heating',ax=ax2)
sb.lineplot(winter.Hour,winter.a2noEh,label='Old no el.heating',ax=ax2)


#summervswinter
plt.figure()
sb.lineplot(summer.Hour,summer.a1Eh,label='Summer')
sb.lineplot(winter.Hour,winter.a1Eh,label='Winter')
plt.title('Seasonal change for one category (a1Eh)')


plt.figure()
ax = sb.violinplot(winter.Hour,winter.a2noEh)



#Stationarity: Augmented dickey fuller test
plt.figure()
print("p-value:", adfuller(a1Eh.dropna())[1])   #0.3
plt.plot(a1Eh.dropna()[1:300])
#Differentiation
print("p-value:", adfuller(a1Eh.diff(24).dropna())[1])  #0.0
plt.plot(a1Eh.diff().dropna()[1:300])



#Autocorrelation and partial autocorrelation
#a1Eh
plt.subplot(1,2)
plot_acf(a1Eh.dropna())
plot_acf(a1Eh.diff(periods=12).dropna())
plot_acf(a1Eh.diff(periods=24).dropna())
plot_pacf(a1Eh.diff(periods=24).dropna())
plt.show()

#Chosing orders p,d,q
pdq=(2,1,)

for pdq in ((1,0,1),(1,0,1),(2,1,1),(1,1,1)):
#ARIMA models order= ()
    model = ARIMA(a1Eh.diff(periods=24).dropna(), order=pdq)
    model_fit = model.fit(disp=0)
    summary = model_fit.summary()
    print(summary)


model = ARIMA(a2Eh.dropna(), order=pdq)
model_fit = model.fit(disp=0)
summary = model_fit.summary()
print(summary)




#Arima testing and generation
#30 days of train and 7 of test

mat.set_index('time', inplace=True, drop=True)
mat.index=pd.to_datetime(mat.index)
mat.index = mat.index.astype('O')
data=mat.iloc[0:888,:].dropna()
data_train=data.iloc[0:720,:].dropna()
data_test=data.iloc[720:888,:].dropna()
data_test.index=pd.to_datetime(data_test.index)
plt.figure()
sb.lineplot(x=data_train.index,y=data_train.a1Eh)
sb.lineplot(x=data_test.index,y=data_test.a1Eh)
model = ARIMA(data_train.a1Eh, order=(1, 1, 1))
results = model.fit()
results.plot_predict(1,888)


#autoarima
import pyramid as pm
from sklearn import metrics
import pmdarima

decomposition=seasonal_decompose(data, model='multiplicative')
trend=decomposition.trend
seasonal= decomposition.seasonal
residual= decomposition.resid

model=pmdarima.auto_arima(data_train.a1Eh,start_p=1,start_q=1, max_p=5,max_q=5,
                 start_P=1,start_Q=1, max_P=5,max_Q=5,m=24,
                 seasonal=True,trace=True,d=1,D=1,
                 error_action='warn',suppress_warnings=True,
                 stepwise=True, random_state=20, n_fits=20)
model.summary()
prediction=pd.DataFrame(model.predict(n_periods=168),index=data_test.index)
prediction.columns=['Hourly Prediction']
plt.figure()
sb.lineplot(x=data_train.index,y=data_train.a1Eh,label='Train')
sb.lineplot(x=data_test.index,y=data_test.a1Eh,label='Actual')
sb.lineplot(x=prediction.index,y=prediction['Hourly Prediction'],label='ARIMA (2,1,1)')
plt.legend()




#Pieplot of categories population
labels='EhEv','noEhEv','EhnoEv','noEhnoEv'
antal=[max(antEhEv),max(antnoEhEv),max(antEhnoEv),max(antnoEhnoEv)]
colors= ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.1, 0.1, 0.1, 0.1)
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.show()









#Trash
################################################################################

decomposition = seasonal_decompose(mat.dropna())

def model (serie, order):
    model = ARIMA(serie, order=order)
    model_fit = model.fit(disp=0)
    summary = model_fit.summary()
    acf = autocorrelation_plot(serie)
    plt.show()
    return (model, order, model_fit, summary, acf)


#acfs
for a in ['a1Eh','a2Eh','a1noEh','a2noEh']:
    acf = autocorrelation_plot(mat[a])
    plt.show



mat['a1Ehdiff']=mat.a1Eh-mat.a1Eh.shift(24)

mod=model(mat.a1Ehdiff[1:500].dropna(),(1,0,0))

#### Auomatic category aggregation

categories=('Dwelling','Age','Heating')
categories=[('_A','_h'),('a1','a2'),('11','12')]

def totmodels (list):
    models=(range(list.__len__()))
    for group in list:
      models(list.index(group))=model(group)

    return(models)

def identify(k,category):
    condition=if category.identifier in k;
    return(condition)


def aggregate (data, type, categories):
    aggr=pd.DataFrame()
    if type=='Average':
        average = data.loc[:, data.columns.isin(k for k in data.columns if 'Average' in k)]
        for category in categories:
            groups= itt.combinations(categories)
            for identifier in category.identifier:
            aggr[idstring] = data.loc[:,data.columns.isin(k for k in average.columns if "qualsiasi identifier nel nome")].mean(axis=1)
    if type=='Sum':
        Sum = data.loc[:, data.columns.isin(k for k in data.columns if 'Sum' in k)]
          for i in categories:
    return(aggr)



comb=itt.combinations(('_A','_h','a1','a2'),2)
y = [' '.join(i) for i in comb]
print(y)




def totmat (a,b,c,d):
    series=[a,b,c,d]
    mat=pd.concat(series,axis=1)
    mat.dropna()
    mat.columns=[]
    return (mat)

def totmodels (a,b,c,d)
    models=dict()
    models=[]
    return(models)









#sum of each columns
singlehouse['sum']=singlehouse.sum(axis=0)


tot=pd.concat(average.loc[1],average.loc[3],pd.get_dummies(average))

#plot line
for group in average.columns:
 sb.lineplot(data.Hour,average[group])

sb.lineplot(data.Hour,average.Average_A1v0ba1uEV11)
sb.lineplot(data.Hour,average.Average_h1v0ba1uEV22)

#plot
tot=data.iloc[:,0]
for group in average.columns:
 tot=pd.concat([tot,average[group]],pd.get_dummies(average))


## ARIMA ##
series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
model = ARIMA(series, order=(12,1,0))

#fit
model_fit = model.fit(disp=0)
print(model_fit.summary())

# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())



singlehouse=data.loc[:, data.columns.isin(k for k in average.columns if 'h1' in k)]
doublehouse=data.loc[:, data.columns.isin(k for k in average.columns if 'h2' in k)]
singleapartment=data.loc[:, data.columns.isin(k for k in average.columns if 'A1' in k)]
doubleapartment=data.loc[:, data.columns.isin(k for k in average.columns if 'A2' in k)]

house=data.loc[:, data.columns.isin(k for k in average.columns if 'Average_h' in k)]
apartments=data.loc[:, data.columns.isin(k for k in average.columns if 'Average_A' in k)]
a1=data.loc[:, data.columns.isin(k for k in average.columns if 'a1' in k)]
a2=data.loc[:, data.columns.isin(k for k in average.columns if 'a2' in k)]
a3=data.loc[:, data.columns.isin(k for k in average.columns if 'a3' in k)]
Eh=data.loc[:, data.columns.isin(k for k in average.columns if '11' in k or '21' in k)]
noEh=data.loc[:, data.columns.isin(k for k in average.columns if '12' in k or '22' in k)]
ha1Eh=data.loc[:, data.columns.isin(k for k in average.columns if 'Average_h' in k or if '11' in k or '21' in k or 'a1' in k)].sum(axis=0)
ha2Eh=data.loc[:, data.columns.isin(k for k in average.columns if 'Average_h' in k or if '11' in k or '21' in k or 'a2' in k)].sum(axis=0)
ha3Eh=data.loc[:, data.columns.isin(k for k in average.columns if 'Average_h' in k or if '11' in k or '21' in k or 'a3' in k)].sum(axis=0)
Aa1Eh=data.loc[:, data.columns.isin(k for k in average.columns if 'Average_A' in k or if '11' in k or '21' in k or 'a1' in k)].sum(axis=0)
Aa2Eh=data.loc[:, data.columns.isin(k for k in average.columns if 'Average_A' in k or if '11' in k or '21' in k or 'a2' in k)].sum(axis=0)
Aa3Eh=data.loc[:, data.columns.isin(k for k in average.columns if 'Average_A' in k or if '11' in k or '21' in k or 'a3' in k)].sum(axis=0)
ha1noEh=data.loc[:, data.columns.isin(k for k in average.columns if 'Average_h' in k or if '12' in k or '22' in k or 'a1' in k)].sum(axis=0)
ha2noEh=data.loc[:, data.columns.isin(k for k in average.columns if 'Average_h' in k or if '12' in k or '22' in k or 'a2' in k)].sum(axis=0)
ha3noEh=data.loc[:, data.columns.isin(k for k in average.columns if 'Average_h' in k or if '12' in k or '22' in k or 'a3' in k)].sum(axis=0)
Aa1noEh=data.loc[:, data.columns.isin(k for k in average.columns if 'Average_A' in k or if '12' in k or '22' in k or 'a1' in k)].sum(axis=0)
Aa2noEh=data.loc[:, data.columns.isin(k for k in average.columns if 'Average_A' in k or if '12' in k or '22' in k or 'a2' in k)].sum(axis=0)
Aa3noEh=data.loc[:, data.columns.isin(k for k in average.columns if 'Average_A' in k or if '12' in k or '22' in k or 'a3' in k)].sum(axis=0)
