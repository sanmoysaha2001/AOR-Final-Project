import numpy as np

# Table 1: Locations, demands, service info
# V, x, y, L, U, P, Demands (Type 1, 2, 3)
table1_data = [
    {"id": 1, "name": "Distribution Center (Start)", "x": 250, "y": 250, "L": 0, "U": 0, "P": 2, "d": [0, 0, 0]},
    {"id": 2, "name": "Dealer 2", "x": 388, "y": 222, "L": 1, "U": 1e9, "P": 3, "d": [1, 1, 1]},
    {"id": 3, "name": "Dealer 3", "x": 364, "y": 408, "L": 1, "U": 1e9, "P": 2, "d": [2, 2, 0]},
    {"id": 4, "name": "Dealer 4", "x": 46, "y": 219, "L": 1, "U": 1e9, "P": 2, "d": [2, 0, 0]},
    {"id": 5, "name": "Dealer 5", "x": 340, "y": 278, "L": 1, "U": 1e9, "P": 3, "d": [2, 0, 0]},
    {"id": 6, "name": "Dealer 6", "x": 388, "y": 360, "L": 1, "U": 1e9, "P": 2, "d": [1, 1, 1]},
    {"id": 7, "name": "Dealer 7", "x": 191, "y": 394, "L": 1, "U": 1e9, "P": 3, "d": [0, 1, 2]},
    {"id": 8, "name": "Dealer 8", "x": 134, "y": 140, "L": 1, "U": 1e9, "P": 3, "d": [2, 0, 0]},
    {"id": 9, "name": "Dealer 9", "x": 85, "y": 71, "L": 1, "U": 1e9, "P": 2, "d": [0, 2, 1]},
    {"id": 10, "name": "Dealer 10", "x": 39, "y": 248, "L": 1, "U": 1e9, "P": 2, "d": [2, 1, 1]},
    {"id": 11, "name": "Dealer 11", "x": 155, "y": 210, "L": 1, "U": 1e9, "P": 2, "d": [1, 1, 0]},
    {"id": 12, "name": "Distribution Center (End)", "x": 250, "y": 250, "L": 0, "U": 1e9, "P": 1, "d": [0, 0, 0]}
]

# Table 2: Time matrix (1-based index 1 to 12)
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

# Table 3: Loading patterns (Type 1, 2, 3)
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

# Let's verify Route 1 and Route 2 reported in Section 6 and Figure 4:
# Route 1: Depot -> 5 -> 2 -> 6 -> 3 -> 7 -> Depot
# Nodes: 1 (depot) -> 5 -> 2 -> 6 -> 3 -> 7 -> 12 (depot)
r1_nodes = [1, 5, 2, 6, 3, 7, 12]
cost1 = 0
for i in range(len(r1_nodes)-1):
    c = time_matrix[r1_nodes[i]-1, r1_nodes[i+1]-1]
    cost1 += c
    print(f"{r1_nodes[i]} -> {r1_nodes[i+1]}: {c}")
print(f"Total cost Route 1: {cost1} (Paper reports: 690)")

# Demands of Route 1:
d1 = np.zeros(3)
for n in [5, 2, 6, 3, 7]:
    d1 += table1_data[n-1]["d"]
print(f"Demands Route 1: {d1}")

# Check Pattern 1 for Route 1: [6, 5, 4]. Sum = 15. Loaded = 6+5+4 = 15. Ratio = 1.0!
print("Pattern 1 cap:", loading_patterns[0]["cap"])

# Route 2: Depot -> 8 -> 9 -> 4 -> 10 -> 11 -> Depot
# Wait, let's check Fig 4:
# Depot -> 11 -> 10 -> 4 -> ... or Depot -> 8 -> 9 -> 4 -> 10 -> 11 -> Depot?
# In paper text page 12/13: "The second route uses loading pattern No. 7, distributing finished vehicles to dealers 8, 9, 4, 10, and 11 with a loading ratio of 50% (8/16)... transport cost 653"
# Let's test both orders:
for order in [[1, 8, 9, 4, 10, 11, 12], [1, 11, 10, 4, 9, 8, 12], [1, 8, 9, 10, 4, 11, 12]]:
    cost = sum(time_matrix[order[i]-1, order[i+1]-1] for i in range(len(order)-1))
    print(f"Order {order}: cost = {cost}")

d2 = sum(np.array(table1_data[n-1]["d"]) for n in [8, 9, 4, 10, 11])
print(f"Demands Route 2: {d2}, Total cars = {sum(d2)}")
print("Pattern 7 cap:", loading_patterns[6]["cap"], "Total:", sum(loading_patterns[6]["cap"]))
print("Loading ratio Route 2 with Pattern 7:", sum(d2) / sum(loading_patterns[6]["cap"]))
