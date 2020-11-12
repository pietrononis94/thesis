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

size=(7,5)
plt.style.use('seaborn')
sb.set_style("darkgrid")
color_list = ['blue','red','green']

#IMPORT

df=pd.read_csv('Data kategorier.csv',skiprows=2,index_col=['READ_TIME_Date'])
df.index = pd.to_datetime(df.index)

######## CLUSTERING DAILY ON ONE SERIE
serie=df.Average_h2vmba2uEV21

dfserie=pd.concat([df.Hour,serie],axis=1)
dfserie = dfserie.astype(np.float).fillna(method='bfill')
df_uci_hourly = dfserie.resample('H').sum()
df_uci_hourly.index = df_uci_hourly.index.date
df_uci_pivot = df_uci_hourly.pivot(columns='Hour')
df_uci_pivot = df_uci_pivot.Average_h2vmba2uEV21.dropna()


#df_uci_pivot.T.plot(figsize=(13,8), legend=False, color='blue', alpha=0.02)
sillhoute_scores = [0.3,0.3]
n_cluster_list = np.arange(2, 31).astype(int)

X = df_uci_pivot.values.copy()

# Very important to scale!
sc = MinMaxScaler()
X = sc.fit_transform(X)
df_uci_pivot
for n_cluster in n_cluster_list:
	kmeans = KMeans(n_clusters=n_cluster)
	cluster_found = kmeans.fit_predict(X)
	sillhoute_scores.append(silhouette_score(X, kmeans.labels_))
plt.figure()
ax = plt.plot(sillhoute_scores)


kmeans = KMeans(n_clusters=5)
cluster_found = kmeans.fit_predict(X)
cluster_found_sr = pd.Series(cluster_found, name='cluster')
df_uci_pivot = df_uci_pivot.set_index(cluster_found_sr, append=True)

fig, ax= plt.subplots(1,1, figsize=(10,5))
color_list = ['blue','red','green','orange','black']
names=['week','weekend','holidays']
cluster_values = sorted(df_uci_pivot.index.get_level_values('cluster').unique())

for cluster, color in zip(cluster_values, color_list):
    df_uci_pivot.xs(cluster, level=1).T.plot(
        ax=ax, legend=False, alpha=0.1, color=color, label= f'Cluster {cluster}'
        )
    df_uci_pivot.xs(cluster, level=1).median().plot(
        ax=ax, color=color, alpha=0.9, ls='--',legend=False
    )

ax.set_xticks(np.arange(0,24))
ax.set_ylabel('Consumption [kW]')
ax.set_xlabel('Hour')
#plt.legend()
#ax.get_legend().remove()


from sklearn.manifold import TSNE
import matplotlib.colors

tsne = TSNE()
results_tsne = tsne.fit_transform(X)

cmap = matplotlib.colors.LinearSegmentedColormap.from_list(cluster_values, color_list)

plt.figure()
plt.scatter(results_tsne[:,0], results_tsne[:,1],
    c=df_uci_pivot.index.get_level_values('cluster'),
    cmap=cmap,
    alpha=0.6,
    )


#df_uci_pivot['week'] = pd.to_datetime(df_uci_pivot.index.get_level_values(0)).strftime('%W')
#df['week']=df.index.strftime('%W')
#dailymean=df_uci_pivot.iloc[0:-1].mean(axis=1)
#dailymean=pd.concat([dailymean[-50:],dailymean])
#dailymean=pd.concat([dailymean[365:],dailymean.dropna()])
#df_uci_pivot['rollingmean']=dailymean.dropna().rolling(window = 50).mean()
#df_uci_pivot['rollingmean']=dailymean[-50:50].dropna().rolling(window = 50).mean()


decomposition = seasonal_decompose(serie.dropna(),model='additive',period=24)
decomptrend=seasonal_decompose(decomposition.trend.dropna(),model='additive',period=168)
plt.figure()
ax = plt.subplot(111)
plt.scatter(pd.to_datetime(df_uci_pivot.index.get_level_values(0)),decomptrend.trend.resample('D').mean(),c=df_uci_pivot.index.get_level_values('cluster'), cmap=cmap,alpha=0.5)
ax.set_xlim((pd.to_datetime('01-01-2017'),pd.to_datetime('31-12-2017')))
plt.ylabel("Consumption [kW]")
ax1 = ax.twiny()
plt.scatter(pd.to_datetime(df_uci_pivot.index.get_level_values(0)),decomptrend.trend.resample('D').mean(),c=df_uci_pivot.index.get_level_values('cluster'), cmap=cmap,alpha=0.5)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%W"))
ax1.set_xlim((pd.to_datetime('01-01-2017'),pd.to_datetime('31-12-2017')))
ax1.set_xlabel('Week')
plt.show()

######## CLUSTERING MONTHLY ON ONE ALL SERIES

#monthlyall=df.loc[:, df.columns.isin(k for k in df.columns if 'Average')]
