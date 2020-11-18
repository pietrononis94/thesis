using Plots, CSV, JuMP, Gurobi, Printf, DataFrames, Tables

Regions=collect(1:2)
Years=collect(1:4)
Periods=collect(1:8759)
Categories=collect(1:17)
Days=collect(1:2917)*3


## Prices
DK1=CSV.read("pricesDK1.csv", delim=",")
DK2=CSV.read("pricesDK2.csv", delim=",")
DK1=dropmissing(DK1)
DK2=dropmissing(DK2)
Prices=[DK1,DK2]
λ=zeros(size(Years)[1],size(Regions)[1],size(Periods)[1])
for r in Regions
    for y in Years
        for t in Periods
            λ[y,r,t]=Prices[r][t,y]
        end
    end
end
max=ones(size(Periods)[1],size(Categories)[1],size(Years)[1])*3




## HEATING FLEX1
FixH1=CSV.read("EVCategoriesFix.csv", delim=",")
FlexH1=CSV.read("EHCategoriesFlex.csv", delim=",")
Ps=FlexH1
FixH1=dropmissing(Fix)
FlexH1=dropmissing(Flex)


## HEATING FLEX 2







final=zeros(size(Periods)[1],size(Categories)[1])
for c in Categories
    model_flexF1 = Model(solver = GurobiSolver())
    @variable(model_flexF1, Pf[t=1:8759]>=0)   #load consumption at time t for category i
    @objective(model_flexF1, Min, sum(λ[y,r,t]*Pf[t] for t in Periods))
    @constraint(model_flexF1, Pf[1] <= Ps[8759,c]+Ps[1,c]+Ps[2,c])
    @constraint(model_flexF1, [t in collect(2:8758)], Pf[t]<=Ps[t-1,c]+Ps[t,c]+Ps[t+1,c])
    @constraint(model_flexF1, [t in Periods], Pf[t]<= Pmax[t,c,y]) #Pmax
    @constraint(model_flexF1, [t in Days], sum(Pf[t+i] for i in collect(0:3))==sum(Ps[t+i,c] for i in collect(0:3)))
    solve(model_flexF1)
final[:,c]=getvalue.(Pf)
end
CSV.write("FlexF1.csv",  DataFrame(final), writeheader=false)






## ELECTRIC VEHICLES
Periods=collect(1:24)
BEVtypes=collect(1:20)
PHEVtypes=collect(1:10)
Availability=CSV.read("Driving_patterns_avail.csv")
KM_demand=CSV.read("Driving_patterns.csv")
AvBEVWD=Availability[1:24,2:21]
KMBEVWD=KM_demand[1:24,2:21]
#KMBEVWE=KM_demand[30:54,2:21]
#KMPHEVWD=KM_demand[63:86,2:11]
#KMPHEVWE=KM_demand[90:113,2:11]
#numbBEV=KM_demand[58,1:20]
#numbPHEV=KM_demand[116,1:10]
ηcar=6     #Km/KWh
ChMax=11.1
ChPHEVMax=4
#check not from the paper too old
ηch=0.95
#Tesla model 3 50-75kWh
BsizeBEV=60

#Plug in between 10 and 15kWh
BsizePHEV=10
M=50000   #BigM



#final=zeros(size(Periods)[1],size(Categories)[1])
#for c in Categories
#BEV
    model_EV = Model(solver = GurobiSolver())
    @variable(model_EV, 0<=ChEV[t in Periods, e in BEVtypes]<= AvBEVWD[t,e]*ChMax)   #load consumption at time t for type e
    @variable(model_EV, 0<=SOC[t in Periods, e in BEVtypes]<=360)
#    @variable(model_EV, Av[t=1:24],Bin)
    @objective(model_EV, Min, sum(λ[y,r,t]*ChEV[t,e] for t in Periods for e in BEVtypes))

    @constraint(model_EV, [t in Periods, e in BEVtypes], SOC[1,e]==SOC[24,e]+ChEV[24,e])
    @constraint(model_EV, [t=2:24, e in BEVtypes], SOC[t,e]==SOC[t-1,e]+ChEV[t,e]-KMBEVWD[t,e]/ηcar)
#    @constraint(model_EV, [t in Periods, e in BEVtypes], ChEV[t,e]<= AvBEVWD[t,e]*)
#    @constraint(model_EV, [t in Periods], SOC[t]>=exit*100)   #when going out full battery needed
#    @constraint(model_EV, [t in Periods, e in BEVtypes], KMBEVWD[t,e]/ηcar+AvBEVWD[t,e]*M>=0.1)
#    @constraint(model_EV, [t in Periods, e in BEVtypes], KMBEVWD[t,e]/ηcar+(AvBEVWD[t,e]-1)*M<=0) #Pmax
    solve(model_EV)
#final[:,c]=getvalue.(Pf)
#end
for e in Periods
    for t in BEVtypes
        print(getvalue.(ChEV[t,e]))
    end
    print('\n')
end
print(getvalue.(SOC))
print(getvalue.(Av))
print(λ[1,1,1:24])

total=DataFrame([getvalue.(ChEV),getvalue.(SOC),getvalue.(Av),λ[1,1,1:24]])
CSV.write("EV.csv",  total, writeheader=false)




# Not using heatpumps when consumption higher than 90% of year peak
Pmax=zeros(size(Periods)[1],size(Categories)[1],size(Years)[1])
for y in Years
    for c in Categories
        for t in Periods
            if Fix[t,c] >= 0.99*maximum(Fix[:,c])
                Pmax[t,c,y]=3
            else
                Pmax[t,c,y]=3
            end
        end
    end
end
y=1
r=1
c=1
