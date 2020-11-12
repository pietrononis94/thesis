#Prices analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

prices=pd.read_csv('prices.csv',index_col='DATETIME',decimal=',')
prices.index=pd.to_datetime(prices.index)

pricesDK1=prices.loc[:,prices.columns.isin(k for k in prices.columns if 'DK1' in k)]

#histograms
plt.figure()
for year in pricesDK1.columns:
    sb.distplot(pricesDK1[year],label=year,bins=50)
plt.legend()


pricesDK1=prices.loc[:,prices.columns.isin(k for k in prices.columns if 'DK1' in k)]
pricesDK2=prices.loc[:,prices.columns.isin(k for k in prices.columns if 'DK2' in k)]


pricesallDK1=pd.DataFrame(pricesDK1.iloc[:,0])
pricesallDK1.columns=['Price']
pricesallDK1['Year']=2020
for col in pricesDK1.columns[1:]:
    serie=pd.DataFrame(pricesDK1[col])
    serie.columns=['Price']
    serie['Year']=col[:4]
#    pricesall=pd.concat([pricesall,serie],axis=1)
    pricesallDK1=pricesallDK1.append(serie)

pricesallDK2=pd.DataFrame(pricesDK2.iloc[:,0])
pricesallDK2.columns=['Price']
pricesallDK2['Year']=2020
for col in pricesDK2.columns[1:]:
    serie=pd.DataFrame(pricesDK2[col])
    serie.columns=['Price']
    serie['Year']=col[:4]
#    pricesall=pd.concat([pricesall,serie],axis=1)
    pricesallDK2=pricesallDK2.append(serie)


pricesallDK1['Region']='DK1'
pricesallDK2['Region']='DK2'
pricesall=pricesallDK1.append(pricesallDK2)
pricesall['Year']=pricesall['Year'].astype(np.int64)

#Violin plot
plt.figure(figsize=(10,5))
#sb.violinplot(x="Year", y="Price", data=pricesall)
sb.violinplot(x=pricesall.Year, y="Price", hue="Region", data=pricesall, palette="muted", split=True,xticks=True)
#plt.xticks([2020,2025,2030,2035,2040])

sb.distplot(pricesall.Year)




pd.read_csv('FlexF1.csv')

