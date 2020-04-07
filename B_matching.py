import networkx as nx
from pulp import *
import matplotlib.pyplot as plt
import numpy as np


def B_min_dist(B):
    B_dist = np.copy(B)
    B_new = np.zeros(B_dist.shape)
    glo_n = B_dist.shape[1]
    for k in range(glo_n):
        ind = np.unravel_index(np.argmin(B_dist), B_dist.shape)
        if B_dist[ind] == 0:
            return B_new
        else:
            B_new[ind] = 1
            B_dist[ind[0], :] = 0
            B_dist[:, ind[1]] = 0
    return B_new


def solve_wbm(from_nodes, to_nodes, wt, from_cap, to_cap):
    logging.info("Start B-matching...")
    # A wrapper function that uses pulp to formulate and solve a WBM
    prob = LpProblem("WBM_Problem", LpMaximize)

    # Create The Decision variables
    choices = LpVariable.dicts("e", (from_nodes, to_nodes), 0, 1, LpInteger)

    # Add the objective function
    prob += lpSum([wt[u][v] * choices[u][v]
                   for u in from_nodes
                   for v in to_nodes]), "Total weights of selected edges"

    # Constraint set ensuring that the total from/to each node
    # is less than its capacity
    for u in from_nodes:
        for v in to_nodes:
            prob += lpSum([choices[u][v] for v in to_nodes]) <= from_cap, ""
            prob += lpSum([choices[u][v] for u in from_nodes]) <= to_cap, ""

    # The problem data is written to an .lp file
    prob.writeLP("WBM.lp")

    # The problem is solved using PuLP's choice of Solver
    prob.solve()

    # The status of the solution is printed to the screen
    logging.debug("Status:" + str(LpStatus[prob.status]))
    logging.info("B-matching complete!")
    return prob


def print_solution(prob):
    # Each of the variables is printed with it's resolved optimum value
    for v in prob.variables():
        if v.varValue > 1e-3:
            print(f'{v.name} = {v.varValue}')
    print(f"Sum of wts of selected edges = {round(value(prob.objective), 4)}")


def get_selected_edges(prob):
    selected_from = [v.name.split("_")[1] for v in prob.variables() if v.value() > 1e-3]
    selected_to = [v.name.split("_")[2] for v in prob.variables() if v.value() > 1e-3]

    selected_edges = []
    for su, sv in list(zip(selected_from, selected_to)):
        selected_edges.append((su, sv))
    return selected_edges


def get_bipartite_positions(graph):
    pos = {}
    for i, n in enumerate(graph.nodes()):
        x = 0 if 'u' in n else 1  # u:0, v:1
        pos[n] = (x, i)
    return (pos)


def get_B_from_selected_edge(selected_edges, B):
    for item in selected_edges:
        n, loc = item[0].split(".")
        glo = item[1]
        B[int(n)][int(loc)][int(glo)] = 1
    return B