import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# IMPORT
df=pd.read_csv('Data kategorier.csv',skiprows=2,index_col=['READ_TIME_Date'])[1:][:]
df.index = pd.to_datetime(df.index)

sb.set_style("darkgrid")
# Houses and apartments
dfH=df[df.columns[5:93]]
dfA=df[df.columns[93:]]

condA=['mEV12', 'uEV11', 'mEV11', 'uEV12'] #aparments
condH=['mEV22', 'uEV21', 'mEV21', 'uEV22'] #houses

# Max average active meters
antalA=[data.max() for data in [df.loc[:, df.columns.isin(k for k in df.columns if 'Antal' in k and cond in k)].sum(axis=1) for cond in condA]]
antalA
antalH=[data.max() for data in [df.loc[:, df.columns.isin(k for k in df.columns if 'Antal' in k and cond in k)].sum(axis=1) for cond in condH]]
antalH

# Percentage share of three classes
shH=[k/(sum(antalH)) for k in antalH]
shA=[k/(sum(antalA)) for k in antalA]
shH
shA

#Profiles of
tsA=pd.concat([df.loc[:, df.columns.isin(k for k in df.columns if 'Average' in k and cond in k)].mean(axis=1) for cond in condA],axis=1)
tsA.columns=['EVEH','noEVnoEH','EVnoEH','noEVEH']
tsH=pd.concat([df.loc[:, df.columns.isin(k for k in df.columns if 'Average' in k and cond in k)].mean(axis=1) for cond in condH],axis=1)
tsH.columns=['EVEH','noEVnoEH','EVnoEH','noEVEH']



# EH profile
EHA=df.loc[:, df.columns.isin(k for k in df.columns if 'Average' in k and '12' in k)].sum(axis=1)
noEHA=df.loc[:, df.columns.isin(k for k in df.columns if 'Average' in k and '11' in k)].sum(axis=1)
EHH=df.loc[:, df.columns.isin(k for k in df.columns if 'Average' in k and '22' in k)].sum(axis=1)
noEHH=df.loc[:, df.columns.isin(k for k in df.columns if 'Average' in k and '21' in k)].sum(axis=1)

EH=pd.concat([EHA,noEHA,EHH,noEHH],axis=1)
EH.columns=['EHA','noEHA','EHH','noEHH']

Aheating=EH.EHA-EH.noEHA
Hheating=EH.EHH-EH.noEHH
summerA=Aheating[Aheating.index.month.isin([6,7,8])]
summerH=Hheating[Hheating.index.month.isin([6,7,8])]

#sum of all heating through year
totheatA=Aheating.sum()
totheatH=Hheating.sum()
print('Total heating consumption: \n Apartments', totheatA ,'Kwh \n Houses', totheatH ,'Kwh')

summer=Average['Ho. Summer']
# Taking away summer heating profile all along year
water =pd.Series(Average['Ho. Summer'][k] for k in Hheating.index.hour.to_list())


heat=Hheating-water

heat.plot()
#daily average
def daily(serie):
    average=serie.groupby(serie.index.hour).mean()
    return (average)

Average=pd.concat([daily(Aheating),daily(Hheating),daily(summerA),daily(summerH)],axis=1)
Average.columns=['Apartments','Houses','Ap. Summer','Ho. Summer']


#plots
plt.figure()
sb.lineplot(Average.index,Average.Houses)

plt.figure()
sb.lineplot(Aheating.index.hour,Aheating)
sb.lineplot(Aheating[Aheating.index.month==8].index.hour,Aheating[Aheating.index.month==8])

plt.figure()
sb.lineplot(Hheating.index.hour,Hheating)
sb.lineplot(Hheating[Hheating.index.month==6 and Hheating.index.month==7 and Hheating.index.month==8].index.hour,Hheating[Hheating.index.month==8])


















#save datframe to json
k=df.iloc[:50,:5].to_json(orient='index')
import json
with open('dataframe.json', 'w') as f:
    json.dump(k, f)
#read json
f = open('dataframe.json')
data = json.load(f)
f.close()

data=pd.read_json('dataframe.json', orient='index')




pd.readcsv('Fle')