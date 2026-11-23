"""Reproduce the finite ABP definition and GCN invariance checks.

Run with the project's Python environment from any working directory.
The default JSON output is written beside this script. The finite checks are
implementation checks; the general mathematical claims are proved separately
in theory_review.md.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import random
import sys
import time

import networkx as nx
import torch
import torch_geometric
from torch_geometric.nn import GCNConv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from models.ABP import filter_edges  # noqa: E402


def directed_edges(arcs: list[tuple[int, int]]) -> torch.Tensor:
    if arcs:
        return torch.tensor(arcs, dtype=torch.long).T.contiguous()
    return torch.empty((2, 0), dtype=torch.long)


def exhaustive_definition_check() -> int:
    count = 0
    for node_count in range(2, 5):
        possibilities = list(itertools.combinations(range(node_count), 2))
        for bits in range(1 << len(possibilities)):
            undirected = [
                edge for index, edge in enumerate(possibilities)
                if bits >> index & 1
            ]
            arcs = undirected + [(target, source) for source, target in undirected]
            edges = directed_edges(arcs)
            graph = nx.Graph()
            graph.add_nodes_from(range(node_count))
            graph.add_edges_from(undirected)
            for mask in range(1, 1 << node_count):
                anchors = {
                    vertex for vertex in range(node_count) if mask >> vertex & 1
                }
                projected, _ = filter_edges(
                    edges, torch.ones(edges.shape[1]), node_count,
                    torch.tensor(sorted(anchors)),
                )
                actual = set(map(tuple, projected.T.tolist()))
                expected = set()
                for source, target in arcs:
                    without_receiver = graph.copy()
                    without_receiver.remove_node(target)
                    component = set(nx.node_connected_component(without_receiver, source))
                    if component & (anchors - {target}):
                        expected.add((source, target))
                if actual != expected:
                    raise AssertionError({
                        "nodes": node_count,
                        "edges": undirected,
                        "anchors": sorted(anchors),
                        "actual": sorted(actual),
                        "expected": sorted(expected),
                    })
                count += 1
    return count


def random_graft_gcn_check() -> tuple[int, float]:
    rng = random.Random(111)
    torch.manual_seed(111)
    maximum = 0.0
    for case in range(100):
        node_count = 7
        graph = nx.gnp_random_graph(node_count, 0.3, seed=case)
        undirected = list(graph.edges())
        arcs = undirected + [(target, source) for source, target in undirected]
        edges = directed_edges(arcs)
        weights = torch.ones(edges.shape[1])
        anchors = torch.tensor([0, 1])
        root = rng.randrange(node_count)
        extra = [(root, node_count), (node_count, root)]
        if case % 3 == 1:
            extra += [(node_count, node_count + 1), (node_count + 1, node_count)]
        if case % 3 == 2:
            extra += [
                (node_count, node_count + 1), (node_count + 1, node_count),
                (node_count + 1, node_count + 2), (node_count + 2, node_count + 1),
                (node_count, node_count + 2), (node_count + 2, node_count),
            ]
        augmented_edges = torch.cat([edges, torch.tensor(extra).T], dim=1)
        augmented_weights = torch.ones(augmented_edges.shape[1])
        base_edges, base_weights = filter_edges(edges, weights, node_count, anchors)
        graft_edges, graft_weights = filter_edges(
            augmented_edges, augmented_weights, node_count + 3, anchors,
        )
        features = torch.randn(node_count, 5)
        augmented_features = torch.cat([features, 10 * torch.randn(3, 5)])
        layer_one = GCNConv(5, 4)
        layer_two = GCNConv(4, 3)
        layer_one.eval()
        layer_two.eval()
        base_output = layer_two(
            torch.relu(layer_one(features, base_edges, base_weights)),
            base_edges, base_weights,
        )
        graft_output = layer_two(
            torch.relu(layer_one(augmented_features, graft_edges, graft_weights)),
            graft_edges, graft_weights,
        )[:node_count]
        error = (base_output - graft_output).abs().max().item()
        maximum = max(maximum, error)
        if not torch.allclose(base_output, graft_output, atol=1e-6, rtol=1e-6):
            raise AssertionError({"case": case, "max_logit_difference": error})
    return 100, maximum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).with_suffix(".json"),
    )
    args = parser.parse_args()
    started = time.perf_counter()
    exhaustive_count = exhaustive_definition_check()
    random_count, maximum = random_graft_gcn_check()
    result = {
        "status": "passed",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "exhaustive_graph_anchor_cases": exhaustive_count,
        "random_graft_gcn_cases": random_count,
        "max_logit_difference": maximum,
        "atol": 1e-6,
        "rtol": 1e-6,
        "seed": 111,
        "exhaustive_node_counts": [2, 3, 4],
        "random_original_node_count": 7,
        "random_added_node_count": 3,
        "random_edge_probability": 0.3,
        "internal_patterns": ["no internal edges", "one internal edge", "triangle"],
        "device": "cpu",
        "torch_version": torch.__version__,
        "torch_geometric_version": torch_geometric.__version__,
        "networkx_version": nx.__version__,
        "elapsed_seconds": time.perf_counter() - started,
        "scope": "Finite implementation checks; general proofs are in theory_review.md.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
