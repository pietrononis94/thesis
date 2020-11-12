import pandas as pd
import datetime as dt
import seaborn as sb
import matplotlib.pyplot as plt


#EV difference
EV=pd.read_csv('PEV-Profiles-L1.csv',skiprows=2)

list=[]
for str in EV.iloc[:, 0].to_list():
    list.append(dt.datetime.strptime(str, "%m/%d/%Y %H:%M"))
EV.index=pd.to_datetime(list)
HEV=EV.resample('H').pad()

#from EVEh script
fact=HEV.mean().mean()/diff.mean()
diff2=diff*fact

plt.figure()
for i in HEV.columns[1:]:
 sb.lineplot(HEV.index.hour,HEV[i],alpha=0.01,ci=0)
sb.lineplot(diff2.index.hour,diff2.Mean)


#EH difference

EH=pd.read_csv('allmeters_hourly.csv')


pd.to_datetime((EH['readdate']+' '+EH['hour']),format='%m/%d/%Y %H:%M')

EH['readhour'].T.astype(str)

EH['date']=(EH['readdate']+' '+EH['readhour'])