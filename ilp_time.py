from gurobipy import *
import argparse
import time
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def create_model_ilp(input_params, map_args, cost):
    print("-------------------------------------------------------------------------------------------------------------")
    print("Starting ILP with cost:" + str(cost))
    print("-------------------------------------------------------------------------------------------------------------")
    env = Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()
    m = Model("ilp", env=env)

    # Variables
    n = len(input_params['B']) # number of data items
    N = len(input_params['tau']) # number of tags
    K = input_params['num_clusters'] # number of clusters

    # y[j][k]: 1 if tag tau_j is in descriptor D_k, 0 otherwise
    y = []
    for j in range(N):
        tau = []
        for k in range(K):
            tau.append(m.addVar(vtype=GRB.BINARY, name="y_"+str(j)+","+str(k)))
        y.append(tau)

    # z[i]: 1 if data item x_i is covered, 0 otherwise
    z = []
    for i in range(n):
        z.append(m.addVar(vtype=GRB.BINARY, name="z_"+str(i)))

    # q[i]: number of tags in D_k that describe x_i (variable creation)
    q = []
    for i in range(n):
        q.append(m.addVar(vtype=GRB.INTEGER, name="q_"+str(i)))

    # Constraints:
    # q[i]: number of tags in D_k that describe x_i (variable initialization)
    for i in range(n):
        cluster = 0
        for k in range(K):
            if i in input_params['C'][k]:
                cluster = k
                break
        expr = LinExpr()
        m.addConstr(quicksum(input_params['B'][i][j] * y[j][cluster] for j in range(N)) == q[i])

    # Ensures that at most B tags are used in total
    if map_args.objective == "MAX":
        m.addConstr(quicksum(y[j][k] for j in range(N) for k in range(K)) <= cost)

    # a tag may appear in at most one descriptors
    for j in range(N):
        m.addConstr(quicksum(y[j][k] for k in range(K)) <= 1)

    # if q_i >= 1, we want to set z_i = 1; otherwise (i.e., q_i = 0), zi should be set to 0.
    for i in range(n):
        m.addConstr(q[i] <= N*z[i])
    for i in range(n):
        m.addConstr(N*z[i] <= q[i] + N - 1)

    # solution set covers at least n % of objects
    if map_args.objective == "MIN":
        for k in range(K):
            m.addConstr(quicksum(z[i] for i in input_params['C'][k]) >= (cost*len(input_params['C'][k]))//100)
        #m.addConstr(quicksum(z[i] for i in range(n)) >= (cost*n)//100)

    # Objective:
    # maximize total coverage
    if map_args.objective == "MAX":
        m.setObjective(quicksum(z[i] for i in range(n)), GRB.MAXIMIZE)
    # minimize total number of tags used
    else:
        m.setObjective(quicksum(y[j][k] for j in range(N) for k in range(K)), GRB.MINIMIZE)

    m.update()

    print("-------------------------------------------------------------------------------------------------------------")
    print("Starting optimization")
    print("-------------------------------------------------------------------------------------------------------------")
    start = time.time()
    m.optimize()
    end = time.time()
    print(end-start)
    return(end-start)

if __name__ == '__main__':

    # read input args
    # new_ilp.py B(cost) eta clusters input_file
    parser=argparse.ArgumentParser()
    parser.add_argument('objective', type=str, help = "'MAX' = maximize items covered ,'MIN' = minimize tags used")
    parser.add_argument('start', type=int, help='enter start cost (int for MAX, percent as int for MIN)')
    parser.add_argument('end', type=int, help='enter end cost (same as start)')
    parser.add_argument('step', type=int, help='enter step (between cost/coverage iterations)')
    parser.add_argument('input_file', type=str, help = '.csv file location of dataset')
    map_args=parser.parse_args()

    if not (map_args.objective == 'MIN' or map_args.objective == 'MAX'):
        sys.exit("Objective Functions must be 'MAX' or 'MIN'")

    # store data into dataframe (pandas)
    dataset = pd.read_csv(map_args.input_file)

    # data item id dropped (not needed)
    #dataset = dataset.drop(['E',], axis=1)

    # count number of clusters
    num_clusters = len(dataset.C.unique())

    # create clusters
    C = []
    for k in range(num_clusters):
        C.append(set())

    # put data items into clusters
    for index, row in dataset.iterrows():
        for k in range(num_clusters):
            if row['C'] == k + 1:
                C[k].add(index)
                break

    # cluster column no longer needed
    dataset = dataset.drop(['C'], axis=1)
    # for threat.csv
    #dataset=dataset.drop(['Seq_Id', 'swiss-prot', 'GO:0044419', 'KW-0181', 'GO:0051704', 'KW-1185', 'GO:0009405', 'GO:0005488', 'GO:0005576',
     # 'GO:0009987', 'GO:0090729', 'KW-0800', 'GO:0008152', 'GO:0003824', 'KW-0964'], axis=1)

    # count data items in each cluster
    cluster_sizes = []
    for k in range(num_clusters):
        cluster_sizes.append(len(C[k]))

    # get tags
    tau = dataset.columns.values.tolist()

    # create a matrix of data items and their tags
    B = dataset.to_numpy()

    input_params={'C': C, 'tau': tau, 'dataset': dataset,
                'B': B, 'num_clusters':num_clusters,
                'cluster_sizes': cluster_sizes}

    # create axis (used later to plot results)
    # X: independent variable
    # MAX: tags used
    # MIN: coverage
    X = [i for i in range(map_args.start,map_args.end+1,map_args.step)]

    # Y: dependent variable
    # MAX: coverage
    # MIN: tags used
    Y = []
    for cost in range(map_args.start,map_args.end+1,map_args.step):
        Y.append(create_model_ilp(input_params, map_args, cost))


    # line graph
    plt.plot(X, Y)
    

    # Naming the x-axis, y-axis and the whole graph
    if map_args.objective == 'MAX':
        plt.xlabel("Tags Used")
        plt.ylabel("Time")
    else:
        plt.xlabel("% Coverage")
        plt.ylabel("Time")

    plt.savefig(map_args.input_file[:-4] + '_time.png')