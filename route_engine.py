import networkx as nx

# Create a simple digital road network
G = nx.Graph()

# Road connections
roads = [
    ("A", "B", 2),
    ("B", "C", 2),
    ("C", "D", 2),
    ("A", "E", 3),
    ("E", "D", 3),
    ("B", "E", 2),
]

for start, end, distance in roads:
    G.add_edge(start, end, weight=distance)

source = "A"
destination = "D"

# Normal route
normal_route = nx.shortest_path(
    G,
    source,
    destination,
    weight="weight"
)

normal_distance = nx.shortest_path_length(
    G,
    source,
    destination,
    weight="weight"
)

print("AEGIS-X ROUTE INTELLIGENCE")
print("---------------------------")
print("Normal Route :", " -> ".join(normal_route))
print("Distance     :", normal_distance)

# Simulate an incident blocking B-C
if G.has_edge("B", "C"):
    G.remove_edge("B", "C")

# Find alternative route
alternate_route = nx.shortest_path(
    G,
    source,
    destination,
    weight="weight"
)

alternate_distance = nx.shortest_path_length(
    G,
    source,
    destination,
    weight="weight"
)

print("\nINCIDENT DETECTED")
print("Blocked Road : B-C")

print("\nALTERNATIVE ROUTE")
print("Route         :", " -> ".join(alternate_route))
print("Distance      :", alternate_distance)