import pandas as pd
import seaborn as sb
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import pmdarima
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib.colors
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
import sklearn.metrics as metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import pandas as pd
import tsfel


# IMPORT
df=pd.read_csv('Data kategorier.csv',skiprows=2,index_col=['READ_TIME_Date'])
df.index = pd.to_datetime(df.index)

# Houses class
serie=df.iloc[:,7].dropna()

cfg_file = tsfel.get_features_by_domain(domain='statistical')
X_train = tsfel.time_series_features_extractor(cfg_file, serie, fs=24, window_splitter=True, window_size=720)


# Remove corr features
corr_features = tsfel.correlated_features(X_train)
X_train.drop(corr_features, axis=1, inplace=True)

X_train['months']=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
X_train.index=X_train['months']
X_train.drop(['months'],axis=1)


Kurtosis=X_train['0_Spectral kurtosis']
#SignifFeatKurtosis=X_train['0_Spectral kurtosis']
Skewness=X_train['0_Skewness']
Energy=X_train['0_Absolute energy']
Variance=X_train['0_Variance']



features=['Kurtosis','Skewness','Energy','Variance']
list=[]
categories=df.columns[df.columns.isin(k for k in df.columns if 'Average' in k)]
for category in categories
    serie=df[category].dropna()
    X_train = tsfel.time_series_features_extractor(cfg_file, serie, fs=24, window_splitter=True, window_size=720)
    list.append[X_train]

mo=[]
for m in months
    df=pd.DataFrame()
    for i in len(list)
    df=df.append(list[i].iloc[m],axis=0)
    df.columns=categories
    df.index=features
    mo.append(df)







df=pd.Dataframe(columns=categories)
for category in categories
    X_train = tsfel.time_series_features_extractor(cfg_file, serie, fs=24, window_splitter=True, window_size=720)
    df[category]=X_train.iloc()










feature=pd.DataFrame()
for category in categories:
    serie=df[category]
    X_train = tsfel.time_series_features_extractor(cfg_file, serie, fs=24, window_splitter=True, window_size=720)
    feat.append(X_train)
    for m in months:
    feature[category]=X_trai










categories=df.columns[df.columns.isin(k for k in df.columns if 'Average' in k)]
gb = df.groupby(['month'])
categ = []
feat =[]
for i in gb.indices:
    df = pd.DataFrame(gb.get_group(i))
    categ.append(df)
    for category in categories:
    serie = df[category]
    X_train = tsfel.time_series_features_extractor(cfg_file, serie, fs=24, window_splitter=True, window_size=720)
    features= X_train
    feat.append(features)




# df
dfserie=pd.concat([df.Hour,df.Average_h2vmba2uEV21],axis=1)
dfserie = dfserie.astype(np.float).fillna(method='bfill')
df_uci_hourly = dfserie.resample('H').sum()
df_uci_hourly.index = df_uci_hourly.index.date
df_uci_pivot = df_uci_hourly.pivot(columns='Hour')
df_uci_pivot = df_uci_pivot.dropna()

#df_uci_pivot.T.plot(figsize=(13,8), legend=False, color='blue', alpha=0.02)

sillhoute_scores = []
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
plt.plot(sillhoute_scores)



kmeans = KMeans(n_clusters=3)
cluster_found = kmeans.fit_predict(X)
cluster_found_sr = pd.Series(cluster_found, name='cluster')
df_uci_pivot = df_uci_pivot.set_index(cluster_found_sr, append=True )

fig, ax= plt.subplots(1,1, figsize=(18,10))
color_list = ['blue','red','green']
cluster_values = sorted(df_uci_pivot.index.get_level_values('cluster').unique())

for cluster, color in zip(cluster_values, color_list):
    df_uci_pivot.xs(cluster, level=1).T.plot(
        ax=ax, legend=False, alpha=0.05, color=color, label= f'Cluster {cluster}'
        )
    df_uci_pivot.xs(cluster, level=1).median().plot(
        ax=ax, color=color, alpha=0.9, ls='--'
    )

ax.set_xticks(np.arange(1,25))
ax.set_ylabel('kilowatts')
ax.set_xlabel('hour')
ax.legend()

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
