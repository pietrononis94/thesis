#Import packages
using JuMP
using Gurobi
using Printf
using MathOptFormat
using CSV

#Sets
profiles = collect(1:3)
periods = collect(1:8759)
categories = collect(1:4)
flex_loads = collect(1:1)
flex_loads = collect(1:2)

#Profiles of EH and EV
#EH= CSV.read("C:/Users/pietr/Documents/GitHub/thesis/EHprofile.csv", delim=",")
#EV= CSV.read("C:/Users/pietr/Documents/GitHub/thesis/EVprofile.csv", delim=",")

#Prices
Prices= CSV.read("C:/Users/pietr/Documents/GitHub/thesis/prices.csv", delim=",")

λ=

#Flexibility parameters
F=[1,2,5,24] #hours of lexibility for classes


####### MODEL #########

model_flex = Model(with_optimizer(Gurobi.Optimizer))

@variable(model_flex, 0<=P[h in periods])   #load consumption at time t for category i

@objective(model_flex, Min, sum(λ[t]*P[t] for t in periods))



### EV


### HF1
@constraint

### HF2


# Max flexibility shift
@constraint(model_flex, max_chp_production[t in periods],  <= )

# Total energy delivered has to be the same
@constraint(model_flex, )
