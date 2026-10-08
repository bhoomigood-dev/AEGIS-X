import networkx as nx
import plotly.graph_objects as go

# ---------------------------------
# Create digital road network
# ---------------------------------
G = nx.Graph()

roads = [
    ("A", "B", 1),
    ("B", "C", 1),
    ("C", "D", 1),
    ("A", "E", 3),
    ("E", "D", 3),
    ("B", "E", 4),
]

for start, end, distance in roads:
    G.add_edge(start, end, weight=distance)

# Node positions
pos = {
    "A": (0, 1),
    "B": (1, 2),
    "C": (2, 2),
    "D": (3, 1),
    "E": (1.5, 0),
}

source = "A"
destination = "D"

# ---------------------------------
# Find NORMAL route
# ---------------------------------
normal_route = nx.shortest_path(
    G,
    source,
    destination,
    weight="weight"
)

# ---------------------------------
# Simulate incident on B-C
# ---------------------------------
blocked_road = ("B", "C")

if G.has_edge(*blocked_road):
    G.remove_edge(*blocked_road)

# ---------------------------------
# Find ALTERNATIVE route
# ---------------------------------
alternative_route = nx.shortest_path(
    G,
    source,
    destination,
    weight="weight"
)

# ---------------------------------
# Create map
# ---------------------------------
fig = go.Figure()

# Draw all roads
for u, v in [
    ("A", "B"),
    ("B", "C"),
    ("C", "D"),
    ("A", "E"),
    ("E", "D"),
    ("B", "E"),
]:

    # Keep blocked road separate
    if {u, v} == set(blocked_road):
        continue

    fig.add_trace(
        go.Scatter(
            x=[pos[u][0], pos[v][0]],
            y=[pos[u][1], pos[v][1]],
            mode="lines",
            line=dict(width=4),
            showlegend=False,
        )
    )

# ---------------------------------
# Highlight NORMAL route
# ---------------------------------
for i in range(len(normal_route) - 1):

    u = normal_route[i]
    v = normal_route[i + 1]

    fig.add_trace(
        go.Scatter(
            x=[pos[u][0], pos[v][0]],
            y=[pos[u][1], pos[v][1]],
            mode="lines",
            line=dict(width=8),
            name="Normal Route",
        )
    )

# ---------------------------------
# Highlight BLOCKED road
# ---------------------------------
u, v = blocked_road

fig.add_trace(
    go.Scatter(
        x=[pos[u][0], pos[v][0]],
        y=[pos[u][1], pos[v][1]],
        mode="lines",
        line=dict(width=10, dash="dash"),
        name="Incident / Blocked Road",
    )
)

# ---------------------------------
# Highlight ALTERNATIVE route
# ---------------------------------
for i in range(len(alternative_route) - 1):

    u = alternative_route[i]
    v = alternative_route[i + 1]

    fig.add_trace(
        go.Scatter(
            x=[pos[u][0], pos[v][0]],
            y=[pos[u][1], pos[v][1]],
            mode="lines",
            line=dict(width=8),
            name="Alternative Route",
        )
    )

# ---------------------------------
# Draw road nodes
# ---------------------------------
fig.add_trace(
    go.Scatter(
        x=[pos[n][0] for n in pos],
        y=[pos[n][1] for n in pos],
        mode="markers+text",
        text=list(pos.keys()),
        textposition="top center",
        marker=dict(size=22),
        showlegend=False,
    )
)

# ---------------------------------
# Layout
# ---------------------------------
fig.update_layout(
    title="AEGIS-X Geographic Intelligence",
    xaxis=dict(showgrid=False, visible=False),
    yaxis=dict(showgrid=False, visible=False),
    template="plotly_white",
    height=600,
)

# ---------------------------------
# Save
# ---------------------------------
fig.write_html("aeGIS_map.html")

print("AEGIS-X MAP READY")
print("-----------------")
print("Normal Route     :", " -> ".join(normal_route))
print("Incident         : B-C road blocked")
print("Alternative Route:", " -> ".join(alternative_route))
print("Map saved as     : aeGIS_map.html")