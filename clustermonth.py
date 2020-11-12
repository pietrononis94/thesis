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
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
#import sklearn.metrics as metrics
import tsfel
from sklearn.metrics import mean_squared_error
import sklearn.metrics as metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.colors
#IMPORT

df=pd.read_csv('Data kategorier.csv',skiprows=2,index_col=['READ_TIME_Date'])
df.index = pd.to_datetime(df.index)


plt.style.use('seaborn')
sb.set_style('darkgrid')
########### Creation of Average df ########
AvgInClasses=df.loc[:,df.columns.isin(k for k in df.columns if 'Average' in k )]
AvgInClasses=AvgInClasses.drop(columns=['Average_huser', 'Average_lejligheder'])
#AvgInClasses.Average_huser[AvgInClasses.index.month==1].plot(style='k.')
# Retrieves a pre-defined feature configuration file to extract all available features
     #  sb.lineplot(AvgInClasses.index[AvgInClasses.index.month==1],AvgInClasses.Average_huser[AvgInClasses.index.month==1])
groups=AvgInClasses.columns





##############clustering for each month###########
size=(7,5)
plt.style.use('seaborn')
sb.set_style("darkgrid")


months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
Avg = dict((months[x], pd.DataFrame()) for x in range(12))
Avg['ncluster']=np.repeat(0,12)
clustermatrix=pd.DataFrame()
#clustermatrix.columns=months
for m in months:
	Avg[m]=AvgInClasses[AvgInClasses.index.month == months.index(m)+1]
	Avg[m]=Avg[m].astype(np.float).fillna(method='bfill')
	X=Avg[m].dropna(axis=1).T.values.copy()
	sillhoute_scores = []
	n_cluster_list = np.arange(2, 20).astype(int)
	for n_cluster in n_cluster_list:
		kmeans = KMeans(n_clusters=n_cluster)
		cluster_found=kmeans.fit_predict(X)
		sillhoute_scores.append(silhouette_score(X, kmeans.labels_))
	#plt.figure()
	#plt.plot(n_cluster_list, sillhoute_scores)
	kmeans = KMeans(n_clusters=n_cluster_list[sillhoute_scores.index(max(sillhoute_scores))])
	Avg['ncluster'][months.index(m)]=n_cluster_list[sillhoute_scores.index(max(sillhoute_scores))]
	cluster_found = kmeans.fit_predict(X)
	cluster_found_sr = pd.Series(cluster_found, name='cluster')
	clustermatrix = pd.concat([clustermatrix,cluster_found_sr],axis=1)
	df_uci_pivot = Avg[m].dropna(axis=1).T.set_index(cluster_found_sr, append=True)
	fig, ax = plt.subplots(1, 1, figsize=(18, 10))
	color_list = ['blue', 'red', 'green']
	cluster_values = sorted(df_uci_pivot.index.get_level_values('cluster').unique())
	for cluster, color in zip(cluster_values, color_list):
		df_uci_pivot.xs(cluster, level=1).T.plot(ax=ax, legend=False, alpha=0.05, color=color, label=f'Cluster {cluster}')
		df_uci_pivot.xs(cluster, level=1).median().plot(ax=ax, color=color, alpha=0.9, ls='--')
	ax.set_ylabel('KWh consumption')
	ax.set_xlabel('%s hourly consumption' % m)

	tsne = TSNE()
	results_tsne = tsne.fit_transform(X)

	cmap = matplotlib.colors.LinearSegmentedColormap.from_list(cluster_values, color_list)

	plt.figure()
	plt.scatter(results_tsne[:,0], results_tsne[:,1],
		c=df_uci_pivot.index.get_level_values('cluster'),
		cmap=cmap,
		alpha=0.6,
		)


	df_uci_pivot['week'] = pd.to_datetime(df_uci_pivot.index.get_level_values(0)).strftime('%W')
	df['week']=df.index.strftime('%W')
	dailymean=df_uci_pivot.iloc[0:-1].mean(axis=1)
	df_uci_pivot['rollingmean']=dailymean.dropna().rolling(window = 50).mean()

	plt.figure()
	#plt.plot(df.week[1:],rolling_mean,label='30 days moving average consumption')
	plt.scatter(pd.to_datetime(df_uci_pivot.index.get_level_values(0)), df_uci_pivot.rollingmean, c=df_uci_pivot.index.get_level_values('cluster'), cmap=cmap,alpha=0.6)
	ax.set_xticks(np.arange(1,25))
	ax.set_ylabel('kiloWatts')
	plt.title('30 days moving average consumption')
	plt.annotate('',xy=(0,1,),xytext=(0,1),fontsize=10)
	ax.legend()
# ax.set_xticks(np.arange(1,25))
#df_uci_hourly = dfserie.resample('H').sum()
#df_uci_hourly.index = df_uci_hourly.index.date
#df_uci_pivot = df_uci_hourly.pivot(columns='Hour')
#df_uci_pivot = df_uci_pivot.dropna()
#df_uci_pivot.T.plot(figsize=(13,8), legend=False, color='blue', alpha=0.02)


plt.figure(figsize=(5,5))
plt.imshow(clustermatrix,cmap='RdYlBu')
sb.heatmap(clustermatrix)




kmeans = KMeans(n_clusters=n_cluster_list[sillhoute_scores.index(max(sillhoute_scores))])
cluster_found = kmeans.fit_predict(X2)
cluster_found_sr = pd.Series(cluster_found, name='cluster')
df_uci_pivot = Avg['Aug'].dropna(axis=1).T.set_index(cluster_found_sr, append=True )

fig, ax= plt.subplots(1,1, figsize=(18,10))
color_list = ['blue','red','green']
cluster_values = sorted(df_uci_pivot.index.get_level_values('cluster').unique())

for cluster, color in zip(cluster_values, color_list):
    df_uci_pivot.xs(cluster, level=1).T.plot(ax=ax, legend=False, alpha=0.05, color=color, label= f'Cluster {cluster}')
    df_uci_pivot.xs(cluster, level=1).median().plot(ax=ax, color=color, alpha=0.9, ls='--')

#ax.set_xticks(np.arange(1,25))
ax.set_ylabel('KWh consumption')
ax.set_xlabel('%s hourly consumption' %m)



tsne = TSNE()
results_tsne = tsne.fit_transform(X)

cmap = matplotlib.colors.LinearSegmentedColormap.from_list(cluster_values, color_list)

plt.figure()
plt.scatter(results_tsne[:,0], results_tsne[:,1],
    c=df_uci_pivot.index.get_level_values('cluster'),
    cmap=cmap,
    alpha=0.6,
    )


df_uci_pivot['week'] = pd.to_datetime(df_uci_pivot.index.get_level_values(0)).strftime('%W')
df['week']=df.index.strftime('%W')
dailymean=df_uci_pivot.iloc[0:-1].mean(axis=1)
df_uci_pivot['rollingmean']=dailymean.dropna().rolling(window = 50).mean()

plt.figure()
#plt.plot(df.week[1:],rolling_mean,label='30 days moving average consumption')
plt.scatter(pd.to_datetime(df_uci_pivot.index.get_level_values(0)), df_uci_pivot.rollingmean, c=df_uci_pivot.index.get_level_values('cluster'), cmap=cmap,alpha=0.6)
ax.set_xticks(np.arange(1,25))
ax.set_ylabel('kiloWatts')
plt.title('30 days moving average consumption')
plt.annotate('',xy=(0,1,),xytext=(0,1),fontsize=10)
ax.legend()


for name in groups:
	sb.lineplot(AvgInClasses.index[AvgInClasses.index.month==1],AvgInClasses.[name][AvgInClasses.index.month==1])(subplots=True, legend=False)

cfg = tsfel.get_features_by_domain()
# Extract features
Features = tsfel.time_series_features_extractor(len(AvgInClasses),AvgInClasses)
