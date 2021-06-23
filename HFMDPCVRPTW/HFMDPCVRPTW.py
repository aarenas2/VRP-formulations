# Imports

import numpy as np
import matplotlib.pyplot as plt
from gurobipy import *
import pandas as pd
import sys
import random
import time

# Seed

rnd = np.random
rnd.seed(0)

# Nodes

cl = 12 # Number of clientes
dep = 2 # Number of depots
n = cl + dep # Number of nodes
days = 6 # Number of days
number_vehicles = 5 # Number of vehicles
number_patterns = 12 # Number of patterns

# Coordinates generation

xc = rnd.rand(cl + dep)*100 # x coordinates
yc = rnd.rand(cl + dep)*50 # y coordinates

# Distance matrix generation

tim = np.zeros([n,n])
for i in range(n):
    for j in range(n):
        tim[i,j] = np.hypot(xc[i]-xc[j],yc[i]-yc[j])

pent = [1,1,1.1,1.1,1.2] # Vehicle time penalty

t = np.zeros([n,n,number_vehicles])
for k in range(number_vehicles):
    t[:,:,k] = tim*pent[k]

penc = [1,1,1.3,1.3,1.5] # Cost of every minute per vehicle

c = np.zeros([n,n,number_vehicles])
for k in range(number_vehicles):
    c[:,:,k] = tim*penc[k]

# Sets

C = [i for i in range(dep,n)] # Set of clients
D = [i for i in range(0,dep)] # Set of deposits
N = D + C # Set of nodes
DELTA = [i for i in range(0,days)] # Set of days
K = [i for i in range(0,number_vehicles)] # Set of vehicles
P = [i for i in range(0,number_patterns)] # Set of patterns

# Graph of map

plt.plot(xc[0],yc[0],c="r",marker="s")
plt.plot(xc[1],yc[1],c="r",marker="s")
plt.scatter(xc[2:],yc[2:],c="b")
for i in N:
    plt.text(xc[i],yc[i],i,fontdict=None)

# Parameters

F = np.loadtxt("visitas.txt") # visit patterns
H = np.loadtxt("assign.txt") # If client can use pattern p
q = [12,12,16,16,20]  # Capacity of vehicle
dem = {i: rnd.randint(1,10) for i in C} # Random demand
s = {i: rnd.randint(10,20) for i in C} # Random time service
s[0]=0 # Adding times of service fron depots
s[1]=0 # Adding times of service fron depots
a = {i: rnd.randint(0,180) for i in C} # Random initial tw
b = {i: a[i]+120 for i in C} # Random initial tw
R = [6,6] # Capacity to attend vehicles
M = 480

# Model

m = Model("HFMDPCVRPTW")
m.setParam("TimeLimit",1000)

# Variables

x = m.addVars(N,N,K,DELTA,vtype=GRB.BINARY,name="x")
T = m.addVars(N,N,K,DELTA, vtype=GRB.CONTINUOUS,lb=0,name="T")
y = m.addVars(C,P,vtype=GRB.BINARY,name="y")
w = m.addVars(D,K,DELTA,vtype=GRB.BINARY,name="w")
f = m.addVars(N,N,K,DELTA, vtype=GRB.CONTINUOUS, lb=0,name="f")

# Objective function

m.modelSense = GRB.MINIMIZE
#m.setObjective(quicksum(T[i,d,k,delta] - T[d,i,k,delta] + t[d,i,k]*x[d,i,k,delta] for i in N for d in D for delta in DELTA for k in K))
m.setObjective(quicksum(t[i,j,k]*x[i,j,k,delta] for i in N for j in N for delta in DELTA for k in K))

# Constraints

## Attend every client according to pattern

m.addConstrs(quicksum(x[c,j,k,delta] for k in K for j in N)==quicksum(F[p,delta]*y[c,p] for p in P) for c in C for delta in DELTA);

## Every client must have a valid pattern assigned

m.addConstrs(quicksum(H[p,c]*y[c,p] for p in P) >= 1 for c in C)

## Depot assignment

m.addConstrs(quicksum(x[d,j,k,delta] for j in N)<=w[d,k,delta] for k in K for delta in DELTA for d in D for delta in DELTA)

## One route per depot

m.addConstrs(quicksum(w[d,k,delta] for d in D) <= 1 for k in K for delta in DELTA)

## leaving the depot returning to depot

m.addConstrs(quicksum(x[i,d,k,delta] for i in N)==quicksum(x[d,j,k,delta] for j in N) for k in K for delta in DELTA for d in D)

## Flux

m.addConstrs(quicksum(x[i,c,k,delta] for i in N)-quicksum(x[c,j,k,delta] for j in N)==0 for c in C for k in K for delta in DELTA)

## Depots capacity

m.addConstrs(quicksum(w[d,k,delta] for k in K)<=R[d] for d in D for delta in DELTA)

## Arrive in tw

m.addConstrs(quicksum(T[i,c,k,delta] for i in N for k in K)>=a[c] for delta in DELTA for c in C)

## Arrive before tw - s

m.addConstrs(quicksum(T[i,c,k,delta] for i in N for k in K)<=b[c]-s[c] for delta in DELTA for c in C)

## Start of T

#m.addConstrs(T[d,j,k,delta] >= t[d,j,k]*x[d,j,k,delta] for d in D for j in N for k in K for delta in DELTA)

## Actualize T

#m.addConstrs(quicksum(T[i,c,k,delta] for i in N for k in K) + s[c] + quicksum(t[c,j,k]*x[c,j,k,delta] for k in K) <= 
              #quicksum(T[c,j,k,delta] for j in N for k in K) for c in C for delta in DELTA)
              
m.addConstrs(quicksum(T[i,c,k,delta] for i in N) + t[c,j,k]*x[c,j,k,delta] + s[c] -
             M * (1 - x[c,j,k,delta]) <= T[c,j,k,delta] for c in C for j in N for delta in DELTA for k in K)

#m.addConstrs(quicksum(T[i,c,k,delta] for i in N) + quicksum(t[c,j,k]*x[c,j,k,delta] + s[c] -
             #M * (1 - x[c,j,k,delta]) for j in N) <= T[c,j,k,delta] for c in C for delta in DELTA for k in K)



## Limit of T

m.addConstrs(T[i,j,k,delta] <= M*x[i,j,k,delta] for k in K for i in N for j in N for delta in DELTA)

## Zeros in diagonal

m.addConstr(quicksum(x[i,i,k,delta] for i in N for k in K for delta in DELTA)==0)

## Load start

m.addConstrs(quicksum(f[d,c,k,delta] for d in D for c in C for delta in DELTA) == 0 for k in K)

## Load flow

m.addConstrs(quicksum(f[i,c,k,delta] + dem[c]*x[c,i,k,delta] for i in N) 
             - quicksum(f[c,j,k,delta] for j in N) == 0 for c in C for k in K for delta in DELTA)

## Bound of f and cap of vehic

m.addConstrs(f[i,j,k,delta] <= q[k]*x[i,j,k,delta] for i in N for j in N for k in K for delta in DELTA)



m.optimize()

#m.printAttr("X")

# Retrieving the variables

xaux = m.getAttr('X', x.values())

cont = 0

x = np.zeros([n,n,number_vehicles,days])
for i in range(n):
    for j in range(n):
        for k in range(number_vehicles):
            for delta in range(days):
                if xaux[cont] > 0:
                    x[i,j,k,delta] = xaux[cont]
                    #print("x [",i,",",j,",",k,",",delta,"]",x[i,j,k,delta])
                cont = cont + 1
                
xd1 = x[:,:,:,0]
                
Taux = m.getAttr('X', T.values())

cont = 0

T = np.zeros([n,n,number_vehicles,days])
for i in N:
    for j in N:
        for k in K:
            for delta in DELTA:
                T[i,j,k,delta] = Taux[cont]
                cont = cont + 1

Td1 = T[:,:,:,0]

for c in C:
    print("TW client ",c, " is [", a[c],"-",b[c],"]")
    
# Day zero


for k in K:
    for i in N:
        for j in N:
            if x[i,j,k,0] > 0.5:
                print("x [",i,",",j,",",k,",",1,"]",x[i,j,k,0])
                print("t [",i,",",j,",",k,"]",t[i,j,k])

for i in N:
    print("T[",i,",",0,"] is ", T[i,0])
    
gap = m.MIPGap

print(gap)
    




















