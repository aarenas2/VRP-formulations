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
rnd.seed(10)

# Nodes

cl = 12 # Number of clientes
dep = 2 # Number of depots
n = cl + dep # Number of nodes
days = 6 # Number of days
number_vehicles = 10 # Number of vehicles
number_patterns = 12 # Number of patterns

# Coordinates generation

xc = rnd.rand(cl + dep)*100 # x coordinates
yc = rnd.rand(cl + dep)*50 # y coordinates

# Distance matrix generation

tim = np.zeros([n,n])
for i in range(n):
    for j in range(n):
        tim[i,j] = np.hypot(xc[i]-xc[j],yc[i]-yc[j])

pent = [1,1,1,1,1.1,1.1,1.1,1.1,1.2,1.2] # Vehicle time penalty

t = np.zeros([n,n,number_vehicles])
for k in range(number_vehicles):
    t[:,:,k] = tim*pent[k]

penc = [1,1,1,1,1.3,1.3,1.3,1.3,1.5,1.5] # Cost of every minute per vehicle

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

# Parameters

F = np.loadtxt("visitas.txt") # visit patterns
H = np.loadtxt("assign.txt") # If client can use pattern p
q = [12,12,12,12,16,16,16,16,20,20]  # Capacity of vehicle
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
m.setParam("TimeLimit",60)

# Variables

x = m.addVars(N,N,K,DELTA,vtype=GRB.BINARY,name="x")
T = m.addVars(N,DELTA, vtype=GRB.CONTINUOUS,lb=0,ub=M,name="T")
y = m.addVars(C,P,vtype=GRB.BINARY,name="y")
w = m.addVars(D,K,vtype=GRB.BINARY,name="w")

# Objective function

m.modelSense = GRB.MINIMIZE
m.setObjective(quicksum(c[i,j,k]*x[i,j,k,delta] for i in N for j in N for k in K for delta in DELTA))

# Constraints

display(P)

## Attend every client according to pattern

m.addConstrs(quicksum(x[c,j,k,delta] for k in K for j in N)==quicksum(F[p,delta]*y[c,p] for p in P) for c in C for delta in DELTA);

## Every client must have a valid pattern assigned

m.addConstrs(quicksum(H[p,c]*y[c,p] for p in P)==1 for c in C)

## Capacity of vehicles

m.addConstrs(quicksum(dem[c]*x[c,j,k,delta] for c in C for j in N) <= q[k] for k in K for delta in DELTA)

## Depot assignment

m.addConstrs(quicksum(x[d,j,k,delta] for j in N)<=w[d,k] for k in K for delta in DELTA for d in D)

## leaving the depot returning to depot

m.addConstrs(quicksum(x[i,d,k,delta] for i in N)==quicksum(x[d,j,k,delta] for j in N) for k in K for delta in DELTA for d in D)

## Flux

m.addConstrs(quicksum(x[i,c,k,delta] for i in N)-quicksum(x[c,j,k,delta] for j in N)==0 for c in C for k in K for delta in DELTA)

## Depots capacity

m.addConstrs(quicksum(w[d,k] for k in K)<=R[d] for d in D)

## Time at depot

m.addConstrs(T[d,delta]==0 for d in D for delta in DELTA)

## Arrive in tw

m.addConstrs(T[c,delta]>=a[c] for delta in DELTA for c in C)

## Arrive before tw - s

m.addConstrs(T[c,delta]<=b[c]-s[c] for delta in DELTA for c in C)

## Actualize T

m.addConstrs(T[c,delta]>=(T[i,delta]+quicksum(t[i,c,k]*x[i,c,k,delta] for k in K)+s[i]) - M*(1-quicksum(x[i,c,k,delta] for k in K)) for i in N for c in C for delta in DELTA)

m.optimize()





















