"""Lightweight knowledge graph from the law-rag JSON data."""
import json
from pathlib import Path
from collections import Counter

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = ["Arial Unicode MS", "Noto Sans CJK", "DejaVu Sans"]

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "json"
OUT_PATH = Path(__file__).resolve().parent.parent / "knowledge_graph.html"


def load_all_records():
    records = []
    for f in DATA_DIR.glob("*.json"):
        with open(f, encoding="utf-8") as fh:
            records.extend(json.load(fh))
    return records


def build_graph(records):
    G = nx.Graph()

    # Collect docs
    docs = {}
    for r in records:
        did = r["doc_id"]
        if did not in docs:
            meta = r.get("doc_meta", {})
            docs[did] = {
                "so_hieu": meta.get("so_hieu", did),
                "co_quan": meta.get("co_quan_ban_hanh", ""),
                "ngay": meta.get("ngay_ban_hanh", ""),
            }

    # Add document nodes
    for did, meta in docs.items():
        label = meta["so_hieu"] or did
        G.add_node(did, label=label, node_type="document",
                   co_quan=meta["co_quan"], ngay=meta["ngay"])

    # Gather top doi_tuong (subjects) across all records
    subj_counter = Counter()
    for r in records:
        for s in r.get("doi_tuong", []):
            s_norm = s.strip().lower()
            if len(s_norm) > 2:
                subj_counter[s_norm] += 1

    top_subjects = {s for s, c in subj_counter.most_common(20)}

    # Gather top keywords
    kw_counter = Counter()
    for r in records:
        for k in r.get("keywords", []):
            k_norm = k.strip().lower()
            if len(k_norm) > 2:
                kw_counter[k_norm] += 1
    top_keywords = {k for k, c in kw_counter.most_common(25)}

    # Add subject nodes + edges
    for r in records:
        did = r["doc_id"]
        for s in r.get("doi_tuong", []):
            s_norm = s.strip().lower()
            if s_norm in top_subjects:
                node_id = f"subj:{s_norm}"
                if node_id not in G:
                    G.add_node(node_id, label=s_norm, node_type="subject")
                if not G.has_edge(did, node_id):
                    G.add_edge(did, node_id, weight=1, edge_type="doi_tuong")
                else:
                    G[did][node_id]["weight"] += 1

    # Add keyword nodes + edges
    for r in records:
        did = r["doc_id"]
        for k in r.get("keywords", []):
            k_norm = k.strip().lower()
            if k_norm in top_keywords:
                node_id = f"kw:{k_norm}"
                if node_id not in G:
                    G.add_node(node_id, label=k_norm, node_type="keyword")
                if not G.has_edge(did, node_id):
                    G.add_edge(did, node_id, weight=1, edge_type="keyword")
                else:
                    G[did][node_id]["weight"] += 1

    # Add clause_type distribution edges (doc → clause_type)
    clause_types = Counter()
    for r in records:
        ct = r.get("normative", {}).get("clause_type", "unknown")
        if ct != "unknown":
            clause_types[(r["doc_id"], ct)] += 1

    for (did, ct), count in clause_types.items():
        node_id = f"type:{ct}"
        if node_id not in G:
            G.add_node(node_id, label=ct, node_type="clause_type")
        G.add_edge(did, node_id, weight=count, edge_type="clause_type")

    return G


def draw_matplotlib(G):
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    pos = nx.spring_layout(G, k=2.5, iterations=80, seed=42)

    # Color + size by node type
    color_map = {
        "document": "#e94560",
        "subject": "#0f3460",
        "keyword": "#16213e",
        "clause_type": "#533483",
    }
    node_sizes = {
        "document": 1800,
        "subject": 900,
        "keyword": 600,
        "clause_type": 1200,
    }

    for ntype in ["keyword", "subject", "clause_type", "document"]:
        nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == ntype]
        if not nodes:
            continue
        nx.draw_networkx_nodes(
            G, pos, nodelist=nodes, ax=ax,
            node_color=color_map[ntype],
            node_size=[node_sizes[ntype]] * len(nodes),
            alpha=0.9,
        )

    # Edges
    edge_colors = {
        "doi_tuong": "#e9456088",
        "keyword": "#0f346088",
        "clause_type": "#53348388",
    }
    for etype in edge_colors:
        edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == etype]
        weights = [G[u][v]["weight"] for u, v in edges]
        if edges:
            nx.draw_networkx_edges(
                G, pos, edgelist=edges, ax=ax,
                width=[min(w * 0.5, 4) for w in weights],
                edge_color=edge_colors[etype],
                alpha=0.6,
            )

    # Labels
    labels = {n: d.get("label", n) for n, d in G.nodes(data=True)}
    # Wrap long labels
    for k, v in labels.items():
        if len(v) > 25:
            labels[k] = v[:22] + "…"

    nx.draw_networkx_labels(G, pos, labels, ax=ax,
                            font_size=8, font_color="white",
                            font_weight="bold")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e94560',
               markersize=14, label='Văn bản pháp luật'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#0f3460',
               markersize=10, label='Đối tượng'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#16213e',
               markersize=8, label='Từ khóa'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#533483',
               markersize=12, label='Loại điều khoản'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
              facecolor='#1a1a2e', edgecolor='#e94560', labelcolor='white')

    ax.set_title("Knowledge Graph — Văn bản pháp luật ngân hàng",
                 fontsize=16, color="white", pad=20)
    ax.axis("off")
    plt.tight_layout()
    out = Path(__file__).resolve().parent.parent / "knowledge_graph.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {out}")
    plt.close()
    return out


def draw_pyvis(G):
    """Interactive HTML version."""
    from pyvis.network import Network

    net = Network(height="900px", width="100%", bgcolor="#1a1a2e",
                  font_color="white", notebook=False)
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=200)

    color_map = {
        "document": "#e94560",
        "subject": "#3282b8",
        "keyword": "#0f3460",
        "clause_type": "#533483",
    }
    size_map = {
        "document": 35,
        "subject": 20,
        "keyword": 15,
        "clause_type": 25,
    }

    for node, data in G.nodes(data=True):
        ntype = data.get("node_type", "keyword")
        title_parts = [f"<b>{data.get('label', node)}</b>", f"Type: {ntype}"]
        if ntype == "document":
            title_parts.append(f"Cơ quan: {data.get('co_quan', '')}")
            title_parts.append(f"Ngày: {data.get('ngay', '')}")
        net.add_node(node, label=data.get("label", node),
                     color=color_map.get(ntype, "#999"),
                     size=size_map.get(ntype, 15),
                     title="<br>".join(title_parts))

    edge_color_map = {
        "doi_tuong": "#e9456099",
        "keyword": "#3282b866",
        "clause_type": "#53348399",
    }
    for u, v, data in G.edges(data=True):
        etype = data.get("edge_type", "")
        net.add_edge(u, v,
                     value=data.get("weight", 1),
                     color=edge_color_map.get(etype, "#55555566"),
                     title=f"{etype} (weight: {data.get('weight', 1)})")

    out = str(OUT_PATH)
    net.save_graph(out)
    print(f"Saved interactive: {out}")
    return out


if __name__ == "__main__":
    records = load_all_records()
    print(f"Loaded {len(records)} records from {len(set(r['doc_id'] for r in records))} documents")

    G = build_graph(records)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    draw_matplotlib(G)
    draw_pyvis(G)
