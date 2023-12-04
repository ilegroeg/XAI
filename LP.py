from gurobipy import *
import argparse
import time
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import random as rand

def create_model_lp(input_params, map_args, cost):
    print("-------------------------------------------------------------------------------------------------------------")
    print("Starting ILP with cost:" + str(cost))
    print("-------------------------------------------------------------------------------------------------------------")
    
    env = Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()
    m = Model("lp", env=env)
    
    # Variables
    n = len(input_params['B']) # number of data items
    N = len(input_params['tau']) # number of tags
    K = input_params['num_clusters'] # number of clusters

    # y[j][k]: 1 if tag tau_j is in descriptor D_k, 0 otherwise
    y = []
    for k in range(K):
      tau = []
      for j in range(N):
        tau.append(m.addVar(ub=1.0, name="y_"+str(k)+","+str(j)))
      y.append(tau)

    # z[i]: 1 if data item x_i is covered, 0 otherwise
    z = []
    for i in range(n):
        z.append(m.addVar(ub=1.0, name="z_"+str(i)))
    
    for k in range(K):
      for i in input_params['C'][k]:
        m.addConstr(quicksum(input_params['B'][i][j]*y[k][j] for j in range(N)) >= z[i])

    # Ensures that at most B tags are used in total
    if map_args.objective == "MAX":
        m.addConstr(quicksum(y[k][j] for j in range(N) for k in range(K)) <= cost)
        for k in range(K):
            m.addConstr(quicksum(z[i] for i in input_params['C'][k]) >= (25*len(input_params['C'][k]))//100)
            print(25*len(input_params['C'][k])//100)

    # solution set covers at least n % of objects
    if map_args.objective == "MIN":
        for k in range(K):
            m.addConstr(quicksum(z[i] for i in input_params['C'][k]) >= (cost*len(input_params['C'][k]))//100)

    # a tag may appear in at most one descriptors
    for j in range(N):
        m.addConstr(quicksum(y[k][j] for k in range(K)) <= 1)

    # Objective:
    # maximize total coverage
    if map_args.objective == "MAX":
        m.setObjective(quicksum(z[i] for i in range(n)), GRB.MAXIMIZE)
    # minimize total number of tags used
    else:
        m.setObjective(quicksum(y[k][j] for j in range(N) for k in range(K)), GRB.MINIMIZE)

    m.update()
    m.write('lp1.lp')

    print("-------------------------------------------------------------------------------------------------------------")
    print("Starting optimization")
    print("-------------------------------------------------------------------------------------------------------------")
    start = time.time()
    m.optimize()
    end = time.time()
    print(end-start)
    # Solution Found
    
    if m.status==GRB.Status.OPTIMAL:
        print("-------------------------------------------------------------------------------------------------------------")
        print("Starting rounding")
        print("-------------------------------------------------------------------------------------------------------------")
        start = time.time()
        y_star = []
        for k in range(K):
            star = []
            for j in range(N):
                star.append(y[k][j].X)
            y_star.append(star)
        z_star = []
        for i in range(n):
            z_star.append(z[i].X)
        
        iteration_results = {}
        for iteration in range(map_args.iterations):
            y_new = []
            y_attr = []
            z_new = []
            hits = []
            for k in range(K):
                y_new.append([0]*N)
                y_attr.append([])
                z_new.append([])
                hits.append(0)
            budget = 0

            for j in range(N):
                vals = []
                for k in range(K):
                    vals.append(y_star[k][j])
                
                vals = np.cumsum(vals)
                random_num = rand.random()

                for k in range(K):
                    if random_num < vals[k]:
                        y_new[k][j] = 1
                        budget += 1
                        y_attr[k].append(y[k][j].VarName)
                        break
            for k in range(K):
                for i in input_params['C'][k]:
                    elem = input_params['B'][i]
                    flag = False
                    for j in range(N):
                        if y_new[k][j] == 1 and elem[j] == 1:
                            flag = True
                            break
                    if flag:
                        z_new[k].append(1)
                        hits[k] +=1
                    else:
                        z_new[k].append(0)

            obj_val = sum(hits)
            if budget <= (2 * cost):
                flag = True
                for k in range(K):
                    if hits[k] < .25*len(input_params['C'][k])/8.0:
                        flag = False
                        break
                if flag:
                    attr_selected = 0
                    for k in range(K):
                        attr_selected += len(y_attr[k])
                    iteration_results[iteration] = {"z_new" : z_new, "obj_value" : obj_val, 'y_new' : y_new, 'y_attr': y_attr, 'attr_selected':attr_selected}
        end = time.time()
        print(end-start)
        print("-------------------------------------------------------------------------------------------------------------")
        print("Rounding results")
        print("-------------------------------------------------------------------------------------------------------------")
        obj_value = 0
        curr_high_value = 0
        high_key = -1
        print("# of candidate solutions: ", len(iteration_results.keys()))
        for key in iteration_results:
            obj_value = iteration_results[key]['obj_value']
            print("Attributes selected: ", key, obj_value)
            if obj_value > curr_high_value :
                curr_high_value = obj_value
                high_key = key
        print("LP Rounding 1 Objective value: ", iteration_results[high_key]['obj_value'])
        total_count = 0
        for k in range(K):
            count = 0
            for i in iteration_results[high_key]['z_new'][k]:
                if i == 1:
                    count += 1
            print("Total # of hits in cluster " + str(k+1) + " after Rounding 1: ", count)
            total_count += count
        total_attr = 0
        for k in range(K):
            attr = 0
            for tag in iteration_results[high_key]['y_new'][k]:
                if tag == 1:
                    attr += 1
            print("# of attributes selected in C"+str(k+1)+": ", attr)
            total_attr += attr
        for k in range(K):
            print("Attribute selected in C"+str(k+1)+" after rounding 1: ", iteration_results[high_key]['y_attr'][k])
        if map_args.objective == "MAX":
            return total_count/n
        else:
            return total_attr
    else:
        print('no sol')
        return 0
    
if __name__ == '__main__':

    # read input args
    # new_ilp.py B(cost) eta clusters input_file
    parser=argparse.ArgumentParser()
    parser.add_argument('objective', type=str, help = "'MAX' = maximize items covered ,'MIN' = minimize tags used")
    parser.add_argument('start', type=int, help='enter start cost (int for MAX, percent as int for MIN)')
    parser.add_argument('end', type=int, help='enter end cost (same as start)')
    parser.add_argument('step', type=int, help='enter step (between cost/coverage iterations)')
    parser.add_argument('input_file', type=str, help = '.csv file location of dataset')
    parser.add_argument('iterations', type=int, help = 'iterations for rounding')
    map_args=parser.parse_args()

    if not (map_args.objective == 'MIN' or map_args.objective == 'MAX'):
        sys.exit("Objective Functions must be 'MAX' or 'MIN'")

    # store data into dataframe (pandas)
    dataset = pd.read_csv(map_args.input_file)

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

    # data item id dropped (not needed)
    #dataset = dataset.drop(['E',], axis=1)
    # for threat.csv
    dataset=dataset.drop(['Seq_Id', 'swiss-prot', 'GO:0044419', 'KW-0181', 'GO:0051704', 'KW-1185', 'GO:0009405', 'GO:0005488', 'GO:0005576',
     'GO:0009987', 'GO:0090729', 'KW-0800', 'GO:0008152', 'GO:0003824', 'KW-0964'], axis=1)

    # count data items in each cluster
    cluster_sizes = []
    for k in range(num_clusters):
        cluster_sizes.append(len(C[k]))
    print('\ncluster sizes: ', cluster_sizes)
    print(' ')

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
    X = []

    # Y: dependent variable
    # MAX: coverage
    # MIN: tags used
    Y = []
    for cost in range(map_args.start,map_args.end+1,map_args.step):
        X.append(cost)
        runs = []
        for i in range(5):
            runs.append(create_model_lp(input_params, map_args, cost))
        Y.append(np.average(runs))
    
    #barplot
    """
    Xs = []
    for i in range(len(Y)):
        Xs.append([x + (i-1)*.60 for x in X])

    for i in range(len(Y)):
        plt.bar(Xs[i], Y[i], width = 0.50, label='C'+str(i+1))
    
    # line graph
    """
    plt.plot(X, Y)
    

    # Naming the x-axis, y-axis and the whole graph
    if map_args.objective == 'MAX':
        plt.xlabel("Tags Used")
        plt.ylabel("% Coverage")
    else:
        plt.xlabel("% Coverage")
        plt.ylabel("Tags Used")

    if map_args.objective == 'MIN':
        yint = []
        locs, labels = plt.yticks()
        for each in locs:
            yint.append(int(each))
        plt.yticks(yint)

    plt.savefig(map_args.input_file[:-4] + '_lp_results.png')