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



#IMPORT

df=pd.read_csv('Data kategorier.csv',skiprows=2,index_col=['READ_TIME_Date'])
df = df.rename(columns = {'priser euro': 'Prices'}, inplace = False)
df.index = pd.to_datetime(df.index)



######## Class aggregation #########
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
	return(aggreg)#,Antal,Average,Tot)


#4classes based on electric heating and EV
EVEh=aggregate(df,['mEV22','mEV12'],'EVEh')
noEVnoEh=aggregate(df,['uEV21','uEV11'],'noEVnoEh')
EVnoEh=aggregate(df,['mEV21','mEV11'],'EVnoEh')
noEVEh=aggregate(df,['uEV22','uEV12'],'noEVEh')

dfEl=pd.concat([df.iloc[:,:4],EVEh,EVnoEh,noEVEh,noEVnoEh],axis=1)




##########
Mean4classes=pd.concat([df.iloc[:,:4],dfEl.MeanEVEh,dfEl.MeannoEVEh,dfEl.MeannoEVnoEh,dfEl.MeanEVnoEh],axis=1)
Mean4classes.columns=['Hour','day','month','weekday','EVEh','noEVEh','noEVnoEh','EVnoEh']



######### Preliminary Plots #########
dfmax=df.loc[:,df.columns.isin(k for k in df.columns if 'Antal' in k)]
dfmax.max(axis=0).plot()

dfav=df.loc[:,df.columns.isin(k for k in df.columns if 'Average' in k)]
pd.concat([dfav.max(axis=0),dfav.idxmax()],axis=1)


#Plot of differences of mean combination with/without EV Eh
ax = plt.subplot(221)
sb.lineplot(dfEl.Hour,dfEl.MeannoEVnoEh,label='no EV no Eh')
ax = plt.subplot(222)
sb.lineplot(dfEl.Hour,dfEl.MeanEVnoEh,label='With EV no Eh')
ax = plt.subplot(223)
sb.lineplot(dfEl.Hour,dfEl.MeannoEVEh,label='no EV With Eh')
ax = plt.subplot(224)
sb.lineplot(dfEl.Hour,dfEl.MeanEVEh,label='With EV With Eh')

#Seasonal Eh
Mean4classesWinter=Mean4classes[Mean4classes.month==(12 or 1 or 2 or 3)]
Mean4classesSummer=Mean4classes[Mean4classes.month==(6 or 7 or 8 )]
plt.figure()
sb.lineplot(Mean4classesWinter.Hour,Mean4classesWinter.EVEh,label='With EV With Eh Winter')
sb.lineplot(Mean4classesWinter.Hour,Mean4classesWinter.EVnoEh,label='With EV no Eh Winter')
sb.lineplot(Mean4classesWinter.Hour,Mean4classesWinter.noEVnoEh,label='no EV no Eh Winter')
sb.lineplot(Mean4classesWinter.Hour,Mean4classesWinter.noEVEh,label='no EV With Eh Winter')
sb.lineplot(Mean4classesSummer.Hour,Mean4classesSummer.EVEh,label='With EV With Eh Summer')
sb.lineplot(Mean4classesSummer.Hour,Mean4classesSummer.EVnoEh,label='With EV no Eh Summer')
sb.lineplot(Mean4classesSummer.Hour,Mean4classesSummer.noEVnoEh,label='no EV no Eh Summer')
sb.lineplot(Mean4classesSummer.Hour,Mean4classesSummer.noEVEh,label='no EV With Eh Summer')


#Electric heating increasing with winter months
plt.figure()
months=('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
for m in months:
    sb.lineplot(Mean4classes.Hour[Mean4classes.month==months.index(m)+1],Mean4classes.noEVEh,label='no EV With Eh month %s'%m,ci=0)


#Pieplot of categories population
plt.figure()
labels=('EhEv','EhnoEv','noEhEv','noEhnoEv')
antal=[max(total4Classes.NumEVEh),max(total4Classes.NumnoEVEh),max(total4Classes.NumEVnoEh),max(total4Classes.NumnoEVnoEh)]
colors= ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0,0,0,0)
plt.pie(antal, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
patches, texts = plt.pie(antal, colors=colors, shadow=True, startangle=90)
plt.legend(patches, labels, loc="best")
plt.axis('Equal')
plt.show()

#whole year and average
fig=plt.figure(figsize=(10,10))
ax = plt.subplot(1, 1, 1)
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
sb.lineplot(mean.index,mean.noEVnoEh,size=1, data=mean.dropna())
rolling_mean = mean.dropna().rolling(window = 720).mean()
sb.lineplot(mean.index,rolling_mean.noEVnoEh, color="coral", label="30 days average trendline")













########################  Models generation   ###########################

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
	  train=sub.iloc[:int(delta*ratio)].dropna()
	  test=sub.iloc[int(delta*ratio):].dropna()
	  return(sub,train,test)





############### ARIMA ###############

data,train,test=subset(EVEh.MeanEVEh,1000,'weekdays',0.1)

#Autocorrelation and partial autocorrelation
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (24, 6))
plot_acf(MeannoEVnoEh.dropna())
plot_acf(MeannoEVnoEh.diff(periods=12).dropna())
plot_acf(MeannoEVnoEh.diff(periods=24).dropna(),ax=ax1)
plot_pacf(MeannoEVnoEh.diff(periods=24).dropna(),ax=ax2)
plt.show()
#Decomposition
decomposition=seasonal_decompose(Mean4classes.noEVnoEh.dropna(), model='moltiplicative', period=24)
decomposition.plot()
trend = decomposition.trend
seasonal= decomposition.seasonal
residual= decomposition.resid
plt.figure(figsize=(10,10))
ax = plt.subplot(221)
ax.xaxis.set_major_locator(plt.MaxNLocator(6))
plt.plot(MeannoEVnoEh,label='Original',linewidth=0.5)
ax = plt.subplot(222)
ax.xaxis.set_major_locator(plt.MaxNLocator(6))
plt.plot(trend,label='Trend',linewidth=0.5)
ax = plt.subplot(223)
ax.xaxis.set_major_locator(plt.MaxNLocator(6))
plt.plot(seasonal,label='Seasonal',linewidth=0.2)
ax = plt.subplot(224)
ax.xaxis.set_major_locator(plt.MaxNLocator(6))
plt.plot(residual,label='Residual',linewidth=0.5)
plt.legend()

#ARIMA fitting
testAR=test
model=pmdarima.auto_arima(test.EVEh,start_p=1,start_q=1, max_p=4,max_q=4,
                 start_P=1,start_Q=1, max_P=4,max_Q=4,m=24,
                 seasonal=True,trace=True,d=1,D=1,
                 error_action='warn',suppress_warnings=True,
                 stepwise=True, random_state=20, n_fits=10)
model.summary()
predictionAR=pd.DataFrame(model.predict(n_periods=168),index=test.index)
prediction.columns=['Hourly Prediction']
performance=errors(testAR,predictionAR)

#Plot
fig=plt.figure(figsize=(10,10))
ax = plt.subplot(1, 1, 1)
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
sb.lineplot(x=train.index,y=train.EVEh,label='Train')
sb.lineplot(x=test.index,y=test.EVEh,label='Actual')
sb.lineplot(x=prediction.index,y=prediction['Hourly Prediction'],label='ARIMA (1,1,0)(5,1,3)')
plt.legend()





######### LINEAR REGRESSION
X = Mean4classes.Hour.values.reshape(-1, 1)  # values converts it into a numpy array
Y = Mean4classes.EVEh.values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions
plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()
plt.figure()



###############LSTM#################

# load the dataset
dataframe = data.EVEh
dataset = dataframe.values.reshape(len(dataframe),1)
dataset = dataset.astype('float32')


# normalize the dataset
# load the dataset
dataset = data.EVEh.T
dataset = dataset.astype('float32')
dataset = np.array(dataset).reshape(-1,1)
#scaler = MinMaxScaler(feature_range=(0, 1))
#dataset = scaler.fit_transform(np.array(dataset).reshape(-1,1))

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return numpy.array(dataX), numpy.array(dataY)


trainE, testE = train.EVEh.values, test.EVEh.values
trainE, testE = train.EVEh.astype('float32'), test.EVEh.astype('float32')


# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(trainE, look_back)
testX, testY = create_dataset(testE, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
#trainPredict = scaler.inverse_transform(trainPredict)
#trainY = scaler.inverse_transform([trainY])
#testPredict = scaler.inverse_transform(testPredict)
#testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back,:] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(data)-1, :] = testPredict
# plot baseline and predictions
plt.figure()
#plt.plot(scaler.inverse_transform(dataset))
plt.plot(dataset)
#plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()



#Errors dataset
import sklearn.metrics as biffa
def errors(test,predictions):
	RMSE = math.sqrt(biffa.mean_squared_error(test,predictions))
	MAPE = np.mean(np.abs((test - predictions) / predictions)) * 100
	MAE = np.mean(np.abs((test - predictions) / predictions))
	MAX = biffa.max_error(test,predictions)
	error = (test-predictions)
	chart=error.plot()
	hist=error.hist()
	return(RMSE,MAPE,MAE,MAX,chart,hist)

dferrors=pd.concat([errors(testAR,predictionsAR),errors(testLR,predictionsLR),errors(testANN,predictionsANN),],axis=0)



############## Eh for sociodemo ############
#Eh and age
EHa1=aggregate(df,['12','22','a1'],'EHa1')
EHa2=aggregate(df,['12','22','a2'],'EHa2')
EHa3=aggregate(df,['12','22','a3'],'EHa3')
NoEHa1=aggregate(df,['11','21','a1'],'noEHa1')
NoEHa2=aggregate(df,['11','21','a2'],'noEHa2')
NOEHa3=aggregate(df,['11','21','a3'],'noEHa3')

dfEhage=pd.concat([df.iloc[:,:4],EHa1,EHa2,EHa3,NoEHa1,NoEHa2,NOEHa3],axis=1)
dfavEhage=dfEhage.iloc[:,dfEhage.columns.isin(k for k in dfEhage.columns if 'Mean' in k)]
dfavEhage=pd.concat([df.iloc[:,:4],dfavEhage],axis=1)
dfavEhage.groupby(['Hour']).mean().iloc[:,3:].plot()


######### Clustering for different
hourlymean=df.groupby(['Hour']).mean()
hourlymean=hourlymean.iloc[:,hourlymean.columns.isin(k for k in hourlymean.columns if 'Average' in k)]
hourlymean.plot()

#correlation heatmap
plt.figure()
sb.heatmap(hourlymean.corr(), xticklabels=divergence.columns,yticklabels=divergence.columns,vmin=0, vmax=1)


#KL divergence heatmap
def divergence(a,b):
	div=metrics.mutual_info_score(a,b)
	return div

div=df.iloc[:,df.columns.isin(k for k in df.columns if 'Average' in k)].corr(method=divergence)
plt.figure()
sb.heatmap(div, xticklabels=divergence.columns,yticklabels=divergence.columns,vmin=1.5, vmax=2)



#differenza con i


###########

