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


categories=df.iloc[1:,:].dropna(axis=1).columns[df.iloc[1:,:].dropna(axis=1).columns.isin(k for k in df.columns if 'Average' in k and 'huser' not in k and 'lejl' not in k)]
#minmaxscaling
scaler = MinMaxScaler(feature_range=(0, 1))
df[categories] = scaler.fit_transform(df[categories])
X_train = tsfel.time_series_features_extractor(cfg_file, serie, fs=24, window_splitter=True, window_size=720)

#serie=
#X_train = tsfel.time_series_features_extractor(cfg_file, serie, fs=24, window_splitter=True, window_size=720)
corr_features = tsfel.correlated_features(X_train)
list=[]
cfg_file = tsfel.get_features_by_domain()
for category in categories:
    serie=df[category].dropna()
    X_train = tsfel.time_series_features_extractor(cfg_file, serie, fs=24, window_splitter=True, window_size=720)
    X_train.drop(corr_features, axis=1, inplace=True)
    #X_train=scaler.fit_transform(X_train)
    list.append(X_train)

mo=[]
for m in np.arange(12):
    data=pd.DataFrame()
    for i in np.arange(len(list)):
        data=data.append(list[i].iloc[m,:])
    data.index=categories
    mo.append(data)



#Clusters and heatmap
n_cluster_list = np.arange(2, 6).astype(int)
months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
clusters=pd.DataFrame()
for m in np.arange(12):
    dataset=mo[m]#.dropna(axis=1)
    sillhoute_scores = [0.0, 0.0]
    for n_cluster in n_cluster_list:
        kmeans = KMeans(n_clusters=n_cluster)
        cluster_found = kmeans.fit_predict(dataset)
        sillhoute_scores.append(silhouette_score(dataset, kmeans.labels_))
    nclu=sillhoute_scores.index(max(sillhoute_scores))
    kmeans = KMeans(n_clusters=nclu)
    cluster_found = kmeans.fit_predict(dataset)
    cluster_found_sr = pd.Series(cluster_found, name='clusters')
    dataset = dataset.set_index(cluster_found_sr, append=True)
    clusters[months[m]]=cluster_found+1

cat=categories.to_list()
cat=[k.split('_')[1] for k in cat]

clusters.index=cat



plt.figure()
#plt.imshow(clusters,cmap='RdYlBu')
cmap=sb.diverging_palette(0, 256, sep=135,n=256, as_cmap=True)
b=sb.heatmap(clusters, cbar=False,annot=True, yticklabels=True, cmap=cmap)#, annot=True, fmt="d")



# Plot of 2 months
# Choose month
# for m in months
m=3
dataset=mo[m]#.dropna(axis=1)
# K means
kmeans = KMeans(n_clusters=3)
cluster_found = kmeans.fit_predict(dataset)
cluster_found_sr = pd.Series(cluster_found, name='clusters')
dataset = dataset.set_index(cluster_found_sr, append=True)
#Plot
df=pd.read_csv('Data kategorier.csv',skiprows=2,index_col=['READ_TIME_Date'])
df.index = pd.to_datetime(df.index)
data=df[df.month==m]
data=data[categories].T
data.index=dataset.index
fig, ax= plt.subplots(1,1, figsize=(18,10))
color_list = ['blue','red','green','orange','brown']
cluster_values = sorted(data.index.get_level_values('clusters').unique())

for cluster, color in zip(cluster_values, color_list):
    data.xs(cluster, level=1).T.plot(
        ax=ax, legend=False, alpha=0.05, color=color, label= f'Cluster {cluster}'
        )
    data.xs(cluster, level=1).mean().plot(
        ax=ax, color=color, alpha=0.9, ls='--'
    )
ax.set_ylabel('kilowatts')
ax.set_xlabel('')

