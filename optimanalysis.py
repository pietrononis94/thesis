#%%
import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
import calendar
#from pylab import *

#%%
FlexF1=pd.read_csv('FlexF1.csv',)
df=pd.read_csv('Data kategorier.csv',skiprows=2,index_col=['READ_TIME_Date'])
df.index=pd.to_datetime(df.index)
FlexF1.index=df.index[0:8758]

#%%
months= np.arange(1,13)
plt.figure(figsize=(20,20))
for m in months:
    month = FlexF1[FlexF1.index.month==m]
    plt.subplot(3,4,m)
    ax=sb.lineplot(month.index.hour, month[])
    ax.set_title('Month= %s'%calendar.month_name[m])

#%%EV
EV=pd.read_csv('EVweek.csv')
