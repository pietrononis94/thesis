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
import sklearn.metrics as biffa
from scipy import stats
from scipy.stats import norm


#IMPORT
df=pd.read_csv('Data kategorier.csv',skiprows=2,index_col=['READ_TIME_Date'])
df.index = pd.to_datetime(df.index)


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
    Ant=Antal.sum(axis=1)
    Aver=Average.sum(axis=1)
    Sum=Tot.sum(axis=1)
	aggreg=pd.concat([Ant,Aver,Sum],axis=1)
	aggreg.columns=['Antal'+name,'Mean'+name,'Sum'+name]
	return(aggreg)

####### Train and test
def subset(serie,start,timeframe,ratio):
	if timeframe == 'day' :
	    delta=24
	elif timeframe=='weekdays':
		delta=120
		serie=serie[df.weekend != 1]
	elif timeframe == 'week' :
		delta=168
	elif timeframe == 'month' :
		delta=720
        elif timeframe == 'year' :
          delta=serie.shape[0]
	else:
		print('\nwrong time frame, insert: day week month or year\n')
	  sub=serie.iloc[start:start+delta].dropna()
	  train=sub.iloc[:int(delta*(1-ratio))].dropna()
	  test=sub.iloc[int(delta*(1-ratio)):].dropna()
	  return(sub,train,test)

#######  Decomposition
def decomposition(serie):
	decomposition=seasonal_decompose(serie.dropna(), model='moltiplicative', period=24)
	decomposition.plot()
	trend = decomposition.trend
	seasonal= decomposition.seasonal
	residual= decomposition.resid
	plt.figure(figsize=(10,10))
	ax = plt.subplot(221)
	ax.xaxis.set_major_locator(plt.MaxNLocator(6))
	plt.plot(serie,label='Original',linewidth=0.5)
	ax = plt.subplot(222)
	ax.xaxis.set_major_locator(plt.MaxNLocator(6))
	plt.plot(trend,label='Trend',linewidth=0.5)
	ax = plt.subplot(223)
	ax.xaxis.set_major_locator(plt.MaxNLocator(6))
	plt.plot(seasonal,label='Seasonal',linewidth=0.2)
	ax = plt.subplot(224)
	plt.legend()
	ax.xaxis.set_major_locator(plt.MaxNLocator(6))
	plt.plot(residual,label='Residual',linewidth=0.5)
	plt.legend()
	return ax

####### ARIMA
def arima(train,test,refit=False):
	model=pmdarima.auto_arima(train.T,start_p=2,start_q=2, max_p=2,max_q=2,
						 start_P=3,start_Q=3, max_P=3,max_Q=3,m=24,
						 seasonal=True,trace=True,d=1,D=1,
						 error_action='warn',suppress_warnings=True,
						 stepwise=True, random_state=20, n_fits=3)
	model.summary()
	prediction=pd.DataFrame(model.predict(n_periods=len(test)),index=test.index)
	prediction.columns=['HourlyPrediction']
	prediction['Model']='ARIMA'
	prediction['Error']=prediction.HourlyPrediction-test
	model.summary()
	return(prediction,model.summary())

####### LR
def LR(train,test):
	X = train.index.hour.values.reshape(-1, 1)  # values converts it into a numpy array
	Y = train.values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
	linear_regressor = LinearRegression()  # create object for the class
	linear_regressor.fit(X, Y)  # perform linear regression
	Y_pred = linear_regressor.predict(test.index.hour.values.reshape(-1,1))  # make predictions
	prediction=pd.DataFrame(Y_pred, index=test.index)
	prediction.columns = ['HourlyPrediction']
	prediction['Model']='LR'
	prediction['Error']=prediction.HourlyPrediction-test
	return (prediction)

####### PR
def PR(train,test):
	mymodel = np.poly1d(np.polyfit(train.index.hour, train.values, 5))
	prediction = mymodel(test.index.hour.values)
	prediction = pd.DataFrame(prediction, index=test.index)
	prediction.columns = ['HourlyPrediction']
	prediction['Model'] = 'PR'
	prediction['Error'] = prediction.HourlyPrediction - test
	return prediction

####### LSTM
def ANNLSTM(train,test):
	dataset = train.T
	dataset = dataset.astype('float32')
	dataset = np.array(dataset).reshape(-1, 1)

	# convert an array of values into a dataset matrix
	def create_dataset(dataset, look_back=1):
		dataX, dataY = [], []
		for i in range(len(dataset) - look_back - 1):
			a = dataset[i:(i + look_back)]
			dataX.append(a)
			dataY.append(dataset[i + look_back])
		return np.array(dataX), np.array(dataY)

	trainE, testE = train.values, test.values
	trainE, testE = trainE.astype('float32'), testE.astype('float32')

	# reshape into X=t and Y=t+1
	look_back = 1
	trainX, trainY = create_dataset(trainE, look_back)
	testX, testY = create_dataset(testE, look_back)

	# reshape input to be [samples, time steps, features]
	trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
	testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

	# create and fit the LSTM network
	model = Sequential()
	model.add(LSTM(4, input_shape=(1, look_back)))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

	# make predictions
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)
	prediction=pd.DataFrame(testPredict,index=test.index[:-2])
	prediction.columns = ['HourlyPrediction']
	prediction['Model']='LSTM'
	prediction['Error']=prediction.HourlyPrediction-test
	return(prediction)

####### Errors
def errors(test,prediction):
	table = pd.DataFrame(columns=['Model', 'RMSE', 'MAPE', 'MAE', 'MAX'])
	for pred in prediction:
		predictions=pred.HourlyPrediction
		test=test.iloc[:pred.shape[0]]
		RMSE = math.sqrt(biffa.mean_squared_error(test,predictions))
		MAPE = np.mean(np.abs((test - predictions) / predictions)) * 100
		MAE = np.mean(np.abs((test - predictions) / predictions))
		MAX = biffa.max_error(test,predictions)
		errors= {'Model':pred.Model[1],'RMSE':RMSE,'MAPE':MAPE,'MAE':MAE,'MAX':MAX}
		table=table.append(errors,ignore_index=True)
	    return(table)




########################  Models use and prediction   ###########################

serie=df.Average_huser

data,train,test=subset(serie,1441,'month',0.234)#last week of march

plt.figure()
test.plot()

####  prediction matrices
predAR=arima(train,test)
predLR=LR(train,test)
predLSTM=ANNLSTM(train,test)
predPR=PR(train,test)
dfpred=pd.concat([predLR,predPR,predAR,predLSTM],axis=0)


####  predictions
plt.figure()
ax=sb.lineplot(x=dfpred.index,y=dfpred.HourlyPrediction,hue=dfpred.Model)
ax=sb.lineplot(x=test.index,y=test,label='real',color='black')
ax.set(xlabel='', ylabel='Household consumption [kW]')
plt.legend()
plt.show()


############PLOTS##########
decomposition = seasonal_decompose(serie.dropna(), model='additive', period=24)
decomptrend=seasonal_decompose(decomposition.trend.dropna(),model='additive',period=168)

plt.figure(figsize=(10,10))
ax1 = plt.subplot(511)
ax1 = sb.lineplot(decomposition.observed.index,decomposition.observed,size=0.01,legend=False)
ax1.set(ylabel='Average House',xlabel='')
ax2 = plt.subplot(512)
ax2 = sb.lineplot(decomposition.observed.index,decomposition.trend,size=0.01,legend=False)
ax2.set(ylabel='trend daily',xlabel='')
ax3 = plt.subplot(513)
ax3 = sb.lineplot(decomptrend.observed.index,decomptrend.trend,size=0.01,legend=False)
ax3.set(xlabel='')
ax4 = plt.subplot(514)
season=decomptrend.seasonal+decomposition.seasonal
ax4 = sb.lineplot(season.index,season,size=0.01,legend=False)
ax4.set(xlabel='')
ax4 = plt.subplot(515)
ax5 = sb.lineplot(decomptrend.observed.index,decomptrend._resid,size=0.01,legend=False)
ax5.set(xlabel='')
plt.show()



plt.figure()
decomptrend.seasonal.iloc[1:500].plot()
plt.show()

plt.figure()
decomptrend.seasonal.hist(bins=50)
plt.show()

plt.figure()
ax=sb.distplot(decomptrend._resid.values,bins=500,norm_hist=True,fit=norm)
#ax.plot(np.arange(-4, +4, 0.001), stats.norm.pdf(np.arange(-4, +4, 0.001)), 'r', lw=2)
ax.set_xlim(-0.05,0.05)
ax.set(xlabel='decomposition residuals[kW]')
plt.show()


#Trend for
serie=df.Average_huser
decomposition = seasonal_decompose(serie.dropna(), model='additive', period=24)
decomptrend=seasonal_decompose(decomposition.trend.dropna(),model='additive',period=168)
plt.figure()
ax=sb.lineplot(decomptrend.observed.index,decomptrend.trend/max(decomptrend.trend.dropna()),size=0.3,legend=False,label='Houses')
serie=df.Average_lejligheder
decomposition = seasonal_decompose(serie.dropna(), model='additive', period=24)
decomptrend=seasonal_decompose(decomposition.trend.dropna(),model='additive',period=168)
#ax2 = plt.twinx()
ax=sb.lineplot(decomptrend.observed.index,decomptrend.trend/max(decomptrend.trend.dropna()),size=0.3,legend=False,color='orange',label='Apartments')
ax.set(xlabel='',ylabel='Trend normalized')
ax.figure.legend(loc='upper center')









#########ACF PACF


serie24=serie.diff(periods=24)

plt.figure(figsize=(7,5))
plot_acf(serie.dropna())

plt.figure(figsize=(10,5))
ax1= plt.subplot(121)
ax1 = plot_acf(serie24.dropna())
ax2 = plt.subplot(122)
ax2 = plot_pacf(serie24.dropna())
plt.show()


plot_acf(serie24.dropna())
plot_acf(serie24168.dropna())
plot_pacf(serie24168.dropna())


data,train,test=subset(serie24168,1000,'month',0.2)
ARIMA1=arima(train,test)
errors(test,[ARIMA1])


serie=df.Average_huser
plot_acf(serie.dropna(), lags=50)
plot_acf(serie.diff(periods=12).dropna())
plot_acf(serie.diff(periods=24).dropna())
plot_acf(serie.diff(periods=24).dropna())
plot_acf(serie.diff(periods=168).dropna())


###### errors distribution plot
plt.figure(figsize=(10, 5))
ax = plt.subplot(121)
for model in ['ARIMA','LR','PR','LSTM']:
	ax=sb.distplot(dfpred.Error[dfpred.Model==model],bins=100,label=model)
	ax.set(ylabel='Frequency', xlabel='Error [kW]')
	ax.set_xlim([-0.3,0.3])
	ax.set_ylim([0,45])
	plt.legend()


######  error VS hourS
ax2 = plt.subplot(122)
ax2=sb.scatterplot(dfpred.index.hour,dfpred.Error,hue=dfpred.Model)
ax2.set(xlabel='Hour', ylabel='Error [kW]')
ax2.legend().texts[0]=(0,0,0)
plt.show()


#table of errors
errorstable=errors(test,[predLR,predPR,predAR,predLSTM])
errorstable.index=errorstable.Model
errorstable=errorstable.drop(['Model'],axis=1)
errorstable.to_latex()


sb.jointplot(serie.index.hour, serie,hue=serie,kind="kde")