import numpy as np
import random
from typing import List, Dict, Tuple, Optional

# Let's test the DAG Split algorithm on the Paper Benchmark
def test_split():
    # 10 dealers + depot
    time_matrix = np.array([
        [0, 141, 195, 206, 94, 176, 156, 160, 243, 211, 103, 0],
        [141, 0, 188, 342, 74, 138, 262, 267, 339, 350, 233, 141],
        [195, 188, 0, 370, 132, 54, 174, 353, 438, 362, 288, 195],
        [206, 342, 370, 0, 300, 370, 227, 118, 153, 30, 109, 206],
        [94, 74, 132, 300, 0, 95, 189, 248, 328, 302, 197, 94],
        [176, 138, 54, 370, 95, 0, 200, 336, 419, 367, 277, 176],
        [156, 262, 174, 227, 189, 200, 0, 260, 340, 211, 187, 156],
        [160, 267, 353, 118, 248, 336, 260, 0, 85, 144, 73, 160],
        [243, 339, 438, 153, 328, 419, 340, 85, 0, 183, 156, 243],
        [211, 350, 362, 30, 302, 367, 211, 144, 183, 0, 122, 211],
        [103, 233, 288, 109, 197, 277, 187, 73, 156, 122, 0, 103],
        [0, 141, 195, 206, 94, 176, 156, 160, 243, 211, 103, 0]
    ])

    # Dealers: IDs 2 to 11
    # Demands:
    demands = {
        2: [1, 1, 1],
        3: [2, 2, 0],
        4: [2, 0, 0],
        5: [2, 0, 0],
        6: [1, 1, 1],
        7: [0, 1, 2],
        8: [2, 0, 0],
        9: [0, 2, 1],
        10: [2, 1, 1],
        11: [1, 1, 0]
    }
    handling_time = {
        2: 3, 3: 2, 4: 2, 5: 3, 6: 2, 7: 3, 8: 3, 9: 2, 10: 2, 11: 2
    }
    time_windows = {
        k: (1, 1e9) for k in range(2, 12)
    }

    loading_patterns = [
        {"id": 1, "cap": [6, 5, 4]},
        {"id": 2, "cap": [5, 4, 4]},
        {"id": 3, "cap": [4, 6, 5]},
        {"id": 4, "cap": [5, 6, 3]},
        {"id": 5, "cap": [4, 5, 6]},
        {"id": 6, "cap": [3, 7, 4]},
        {"id": 7, "cap": [7, 4, 5]},
        {"id": 8, "cap": [6, 4, 5]},
        {"id": 9, "cap": [5, 7, 2]}
    ]

    # Let's test evaluating the paper's exact optimal permutation:
    # Route 1: [5, 2, 6, 3, 7]
    # Route 2: [8, 9, 4, 10, 11]
    perm = [5, 2, 6, 3, 7, 8, 9, 4, 10, 11]

    # Let's write the split function
    def split_chromosome(chromosome, patterns, time_mat, dem, hand_time, tw, max_dur=1e9, alpha=1.0):
        n = len(chromosome)
        # DAG nodes 0 to n
        # dist[j] = min cost to cover first j dealers in chromosome
        dist = [float('inf')] * (n + 1)
        parent = [-1] * (n + 1)
        route_info = {}
        dist[0] = 0

        for i in range(n):
            if dist[i] == float('inf'):
                continue
            
            cum_dem = np.zeros(3, dtype=int)
            route_nodes = []
            
            for j in range(i + 1, n + 1):
                dealer = chromosome[j - 1]
                route_nodes.append(dealer)
                cum_dem += np.array(dem[dealer])
                
                # Check pattern compatibility
                feasible_patterns = []
                for p in patterns:
                    cap = np.array(p["cap"])
                    if np.all(cum_dem <= cap):
                        ratio = np.sum(cum_dem) / np.sum(cap)
                        feasible_patterns.append((p, ratio))
                
                if not feasible_patterns:
                    # Adding more dealers will only increase demand, so break early
                    break
                
                # Pick best pattern
                # If alpha=1.0, minimize travel time, pick pattern that maximizes ratio
                feasible_patterns.sort(key=lambda x: x[1], reverse=True)
                best_pat, best_ratio = feasible_patterns[0]
                
                # Check time windows & travel time
                # Route: 1 -> route_nodes -> 12
                # Note: node 1 in time_mat is index 0; node 12 is index 11; dealer d is index d-1
                t = 0
                prev_idx = 0 # depot start
                curr_time = 0
                travel_cost = 0
                tw_feasible = True
                
                for d in route_nodes:
                    d_idx = d - 1
                    travel_time = time_mat[prev_idx, d_idx]
                    travel_cost += travel_time
                    arrival = curr_time + travel_time
                    L, U = tw[d]
                    if arrival > U:
                        tw_feasible = False
                        break
                    start_serv = max(arrival, L)
                    curr_time = start_serv + hand_time[d]
                    prev_idx = d_idx
                
                if not tw_feasible:
                    continue
                
                # Return to depot
                depot_return_idx = 11
                return_travel = time_mat[prev_idx, depot_return_idx]
                travel_cost += return_travel
                curr_time += return_travel
                
                if curr_time > max_dur:
                    continue
                
                # Objective formulation:
                # Travel time cost f = travel_cost
                # Empty slots cost = sum(pat_cap) - sum(cum_dem)
                empty_slots = np.sum(best_pat["cap"]) - np.sum(cum_dem)
                # Weighted edge cost
                edge_cost = alpha * travel_cost + (1 - alpha) * (empty_slots * 10)
                
                if dist[i] + edge_cost < dist[j]:
                    dist[j] = dist[i] + edge_cost
                    parent[j] = i
                    route_info[(i, j)] = {
                        "dealers": list(route_nodes),
                        "travel_cost": travel_cost,
                        "pattern": best_pat,
                        "demands": cum_dem.copy(),
                        "ratio": best_ratio,
                        "total_duration": curr_time
                    }
        
        if dist[n] == float('inf'):
            return None, float('inf')
        
        # Reconstruct routes
        curr = n
        routes = []
        while curr > 0:
            p = parent[curr]
            routes.append(route_info[(p, curr)])
            curr = p
        routes.reverse()
        return routes, dist[n]

    routes, total_cost = split_chromosome(perm, loading_patterns, time_matrix, demands, handling_time, time_windows, alpha=1.0)
    print(f"Split completed. Number of routes: {len(routes)}, Total Cost: {total_cost}")
    for idx, r in enumerate(routes):
        print(f"Route {idx+1}: Dealers {r['dealers']}, Travel Cost: {r['travel_cost']}, Pattern: {r['pattern']['id']} {r['pattern']['cap']}, Demands: {r['demands']}, Ratio: {r['ratio']:.3f}")

if __name__ == "__main__":
    test_split()
