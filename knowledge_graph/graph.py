"""Knowledge graph construction and visualization using NetworkX and PyVis."""

from __future__ import annotations

import colorsys
import math
from pathlib import Path

import community as community_louvain
import networkx as nx
from pyvis.network import Network


def build_graph(triplets: list[dict]) -> nx.Graph:
    """Build a NetworkX graph from SPO triplets.

    Args:
        triplets: List of dicts with subject, predicate, object, and optional inferred flag.

    Returns:
        A NetworkX Graph with nodes and labeled edges.
    """
    G = nx.Graph()

    for t in triplets:
        subj = t.get("subject", "").strip()
        obj = t.get("object", "").strip()
        pred = t.get("predicate", "").strip()
        inferred = t.get("inferred", False)

        if not subj or not obj:
            continue

        G.add_node(subj)
        G.add_node(obj)
        G.add_edge(subj, obj, label=pred, inferred=inferred)

    return G


def detect_communities(G: nx.Graph) -> dict[str, int]:
    """Detect communities using the Louvain method.

    Returns:
        A dict mapping node name to community id.
    """
    if len(G.nodes()) == 0:
        return {}
    return community_louvain.best_partition(G)


def compute_node_sizes(G: nx.Graph) -> dict[str, float]:
    """Compute node sizes based on degree centrality and betweenness centrality.

    Returns:
        A dict mapping node name to a size value.
    """
    if len(G.nodes()) == 0:
        return {}

    degree = nx.degree_centrality(G)
    try:
        betweenness = nx.betweenness_centrality(G)
    except Exception:
        betweenness = {n: 0 for n in G.nodes()}

    sizes = {}
    for node in G.nodes():
        score = 0.6 * degree.get(node, 0) + 0.4 * betweenness.get(node, 0)
        size = 10 + score * 40
        sizes[node] = size

    return sizes


def _generate_colors(n: int) -> list[str]:
    """Generate n visually distinct colors."""
    colors = []
    for i in range(n):
        hue = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.85)
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colors


def get_graph_stats(G: nx.Graph, communities: dict[str, int]) -> dict:
    """Compute summary statistics for the graph."""
    n_communities = len(set(communities.values())) if communities else 0
    inferred_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("inferred", False))
    extracted_edges = G.number_of_edges() - inferred_edges

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "extracted_edges": extracted_edges,
        "inferred_edges": inferred_edges,
        "communities": n_communities,
    }


def visualize_graph(
    G: nx.Graph,
    communities: dict[str, int],
    sizes: dict[str, float],
    dark_mode: bool = False,
    height: str = "700px",
    width: str = "100%",
) -> str:
    """Create an interactive PyVis HTML visualization of the knowledge graph.

    Args:
        G: The NetworkX graph.
        communities: Mapping of node -> community id.
        sizes: Mapping of node -> display size.
        dark_mode: Whether to use dark background.
        height: Height of the visualization.
        width: Width of the visualization.

    Returns:
        HTML string of the interactive visualization.
    """
    bg_color = "#1a1a2e" if dark_mode else "#ffffff"
    font_color = "#e0e0e0" if dark_mode else "#333333"

    net = Network(
        height=height,
        width=width,
        bgcolor=bg_color,
        font_color=font_color,
        directed=False,
        notebook=False,
    )

    net.barnes_hut(
        gravity=-3000,
        central_gravity=0.3,
        spring_length=150,
        spring_strength=0.05,
        damping=0.09,
    )

    # Generate community colors
    if communities:
        n_communities = len(set(communities.values()))
        colors = _generate_colors(n_communities)
        community_colors = {cid: colors[i] for i, cid in enumerate(sorted(set(communities.values())))}
    else:
        community_colors = {}

    # Add nodes
    for node in G.nodes():
        cid = communities.get(node, 0)
        color = community_colors.get(cid, "#6c757d")
        size = sizes.get(node, 15)
        degree = G.degree(node)
        title = f"<b>{node}</b><br>Community: {cid}<br>Connections: {degree}"

        net.add_node(
            node,
            label=node,
            title=title,
            color=color,
            size=size,
            font={"size": max(8, int(size * 0.8)), "color": font_color},
        )

    # Add edges
    for u, v, data in G.edges(data=True):
        label = data.get("label", "")
        inferred = data.get("inferred", False)

        edge_color = "#aaaaaa" if not dark_mode else "#555555"
        if inferred:
            edge_color = "#ff6b6b" if dark_mode else "#e74c3c"

        net.add_edge(
            u,
            v,
            title=label,
            label=label,
            color=edge_color,
            dashes=inferred,
            width=1 if not inferred else 1.5,
            font={"size": 8, "color": font_color, "strokeWidth": 0},
        )

    # Generate HTML
    html = net.generate_html()

    # Add custom controls overlay
    controls_html = f"""
    <div id="graph-controls" style="
        position: absolute; top: 10px; right: 10px; z-index: 1000;
        background: {'rgba(26,26,46,0.9)' if dark_mode else 'rgba(255,255,255,0.9)'};
        padding: 10px; border-radius: 8px;
        font-family: Arial, sans-serif; font-size: 12px;
        color: {font_color}; border: 1px solid {'#333' if dark_mode else '#ddd'};
    ">
        <div style="margin-bottom:5px"><b>Legend</b></div>
        <div><span style="display:inline-block;width:30px;border-top:2px solid {'#aaa' if not dark_mode else '#555'}"></span> Extracted</div>
        <div><span style="display:inline-block;width:30px;border-top:2px dashed {'#e74c3c' if not dark_mode else '#ff6b6b'}"></span> Inferred</div>
    </div>
    """

    html = html.replace("</body>", controls_html + "</body>")

    return html
