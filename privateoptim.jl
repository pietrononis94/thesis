using Plots, CSV, JuMP, Gurobi, Printf, DataFrames, Tables

Regions=collect(1:2)
Years=collect(1:4)
Periods=collect(1:8759)
Categories=collect(1:17)
Days=collect(1:2917)*3

### Profiles
Fix=CSV.read("EHCategoriesFix.csv", delim=",")
Flex=CSV.read("EHCategoriesFlex.csv", delim=",")
Ps=Flex
Fix=dropmissing(Fix)
Flex=dropmissing(Flex)
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











model_flexF1 = Model(solver = GurobiSolver())
@variable(model_flexF1, Pf[t=1:8759,c=1:17]>=0)   #load consumption at time t for category i
@objective(model_flexF1, Min, sum(λ[y,r,t]*Pf[t,c] for t in Periods))
@constraint(model_flexF1, [c in Categories], Pf[1,c] <= Ps[8759,c]+Ps[1,c]+Ps[2,c])
@constraint(model_flexF1, [t in collect(2:8758), c in Categories], Pf[t,c]<=Ps[t-1,c]+Ps[t,c]+Ps[t+1,c])
@constraint(model_flexF1, [c in Categories], Pf[8759,c] <= Ps[8757,c]+Ps[8758,c]+Ps[8759,c])
#@constraint(model_flexF1, [t in Periods, c in Categories], Pf[t,c]<= 3) #Pmax
#@constraint(model_flexF1, sum(Pf[t,c] for t in Periods)==sum(Ps[t,c] for t in Periods))
solve(model_flexF1)





if termination_status(model_flex) == MOI.OPTIMAL
    for t in periods
        output_line = "$(@sprintf("%.3f",λ[t]))\t"
        print(output_line)
    end
    print('\n')
    for t in periods
        output_line = "$(@sprintf("%.3f",Ps[t]))\t"
        print(output_line)
    end
    print('\n')
    for t in periods
        output_line = "$(@sprintf("%.3f",(value.(Pf[t]))))\t"
        print(output_line)
    end
    @printf "\nObjective value: %0.3f\n\n" objective_value(model_flex)
else
    error("No solution.")
end


plot(getvalue.(Pf).data)
plot!(Ps)
plot!(Pnonflex)
pfin=Pnonflex+Ps*0.7+0.3value.(Pf).data
plot!(pfin)
