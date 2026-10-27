"""Tema 4, Exercitiul 2: predictia vremii cu o retea Markov discreta.

Ruleaza: python exercitiul_2_pgmpy.py
Dependinte: numpy, networkx, matplotlib, pgmpy.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TypeAlias

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.typing import NDArray
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import MarkovNetwork

FloatArray: TypeAlias = NDArray[np.float64]
Configuration: TypeAlias = tuple[int, ...]
WeightedConfiguration: TypeAlias = tuple[Configuration, float]
Edge: TypeAlias = tuple[int, int]

STATES: tuple[str, ...] = ("Soare", "Ploaie", "Nori")


@dataclass(frozen=True)
class ModelConfig:
    """Define all inputs for the Markov random field.

    Attributes:
        station_count: Number of weather stations.
        edges: Undirected edges; tuple order sets the factor argument order.
        unary: Positive unary weights for Soare, Ploaie, Nori.
        pairwise: Positive 3x3 pairwise potential matrix.
    """

    station_count: int
    edges: tuple[Edge, ...]
    unary: FloatArray
    pairwise: FloatArray

    @property
    def names(self) -> tuple[str, ...]:
        """Return station names in their numeric order."""
        return tuple(f"S{i + 1}" for i in range(self.station_count))


def make_config() -> ModelConfig:
    """Create the five-station chain and the potentials from the assignment."""
    return ModelConfig(
        station_count=5,
        edges=((0, 1), (1, 2), (2, 3), (3, 4)),
        unary=np.array([0.5, 0.3, 0.2], dtype=np.float64),
        pairwise=np.array(
            [[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]],
            dtype=np.float64,
        ),
    )


def make_graph(config: ModelConfig) -> nx.Graph:
    """Construct and validate the connected acyclic station graph.

    Args:
        config: Model definition with station count and edges.

    Returns:
        An undirected NetworkX tree.

    Raises:
        ValueError: If station indices, potentials, or graph are invalid.
    """
    if config.station_count < 1:
        raise ValueError("station_count must be positive")
    if config.unary.shape != (3,) or config.pairwise.shape != (3, 3):
        raise ValueError("Expected 3 unary weights and a 3x3 pairwise matrix")
    if not (np.all(np.isfinite(config.unary)) and np.all(config.unary > 0)):
        raise ValueError("Unary weights must be positive and finite")
    if not np.isclose(config.unary.sum(), 1.0):
        raise ValueError("Unary weights must sum to one")
    if not (np.all(np.isfinite(config.pairwise)) and np.all(config.pairwise > 0)):
        raise ValueError("Pairwise weights must be positive and finite")
    if any(not (0 <= i < config.station_count and 0 <= j < config.station_count)
           for i, j in config.edges):
        raise ValueError("Invalid station index in edges")

    graph = nx.Graph()
    graph.add_nodes_from(config.names)
    graph.add_edges_from((config.names[i], config.names[j]) for i, j in config.edges)
    if not nx.is_tree(graph) or graph.number_of_edges() != len(config.edges):
        raise ValueError("The supplied edges must describe a tree")
    return graph


def draw_graph(graph: nx.Graph, path: str = "ex2_graf.png") -> None:
    """Save and display the station graph for points a) and c).

    Args:
        graph: Graph to visualize.
        path: Output PNG file name.
    """
    positions = {name: (index, 0) for index, name in enumerate(graph.nodes)}
    fig, ax = plt.subplots(figsize=(10, 2.5))
    nx.draw_networkx(
        graph, pos=positions, ax=ax, node_color="#cfe4ff",
        edgecolors="#2870cf", node_size=1200, font_weight="bold",
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.show()
    plt.close(fig)


def make_markov_model(config: ModelConfig) -> MarkovNetwork:
    """Construct a pgmpy MRF with unary and pairwise factors.

    Args:
        config: Validated network parameters.

    Returns:
        A pgmpy MarkovNetwork with 5 unary and 4 pairwise factors.
    """
    model = MarkovNetwork()
    model.add_nodes_from(config.names)
    model.add_edges_from((config.names[i], config.names[j]) for i, j in config.edges)
    for name in config.names:
        model.add_factors(DiscreteFactor([name], [3], config.unary.copy()))
    for i, j in config.edges:
        # IMPORTANT: do not transpose the non-symmetric pairwise potential.
        model.add_factors(
            DiscreteFactor([config.names[i], config.names[j]], [3, 3], config.pairwise.copy())
        )
    model.check_model()
    return model


def weight(configuration: Configuration, config: ModelConfig) -> float:
    """Calculate the unnormalized product of all MRF potentials.

    Args:
        configuration: State indices (0=Soare, 1=Ploaie, 2=Nori).
        config: Model parameters.

    Returns:
        Positive score w(s) before division by Z.
    """
    if len(configuration) != config.station_count or any(s not in (0, 1, 2) for s in configuration):
        raise ValueError("Invalid weather configuration")
    unary_product = float(np.prod(config.unary[list(configuration)]))
    pair_product = 1.0
    for i, j in config.edges:
        pair_product *= float(config.pairwise[configuration[i], configuration[j]])
    return unary_product * pair_product


def enumerate_weights(config: ModelConfig) -> list[WeightedConfiguration]:
    """Enumerate all 3^n configurations and calculate their scores."""
    return [(states, weight(states, config))
            for states in product(range(3), repeat=config.station_count)]


def filter_evidence(
    rows: list[WeightedConfiguration], evidence: dict[int, int], config: ModelConfig
) -> list[WeightedConfiguration]:
    """Keep configurations agreeing with all observed station states.

    Args:
        rows: All configurations and their scores.
        evidence: Station-index -> observed state-index mapping.
        config: Station count, for index validation.

    Returns:
        Configurations consistent with evidence.
    """
    if any(i < 0 or i >= config.station_count or s not in (0, 1, 2)
           for i, s in evidence.items()):
        raise ValueError("Invalid evidence")
    return [(states, score) for states, score in rows
            if all(states[i] == s for i, s in evidence.items())]


def marginal(
    rows: list[WeightedConfiguration], station_index: int, evidence_normalizer: float
) -> FloatArray:
    """Compute P(S_i | evidence) by exact enumeration.

    Args:
        rows: Evidence-compatible configurations and weights.
        station_index: Index of the target station.
        evidence_normalizer: Sum of weights over evidence-compatible rows.

    Returns:
        Probabilities for Soare, Ploaie and Nori, in this order.
    """
    if evidence_normalizer <= 0:
        raise ValueError("Evidence has zero total weight")
    if not rows or not 0 <= station_index < len(rows[0][0]):
        raise ValueError("Invalid station or empty rows")
    return np.array([
        sum(score for states, score in rows if states[station_index] == s) / evidence_normalizer
        for s in range(3)
    ], dtype=np.float64)


def global_map(
    rows: list[WeightedConfiguration], evidence_normalizer: float
) -> tuple[Configuration, float]:
    """Return the joint MAP configuration and conditional probability.

    Args:
        rows: Evidence-compatible configurations.
        evidence_normalizer: Sum of scores in those rows.

    Returns:
        Most probable joint configuration, followed by P(MAP | evidence).
    """
    if not rows or evidence_normalizer <= 0:
        raise ValueError("No valid evidence-compatible configuration")
    states, score = max(rows, key=lambda row: row[1])
    return states, score / evidence_normalizer


def main() -> None:
    """Run assignment points printing results and drawing the graph."""
    config = make_config()

    # a) Draw and validate tree, print neighbors.
    graph = make_graph(config)
    print("a) Arbore:", nx.is_tree(graph))
    for name in config.names:
        print(f"   Vecinii lui {name}: {', '.join(sorted(graph.neighbors(name)))}")
    draw_graph(graph)

    # b) Build the Markov network with 5 unary and 4 pairwise factors.
    model = make_markov_model(config)
    print("\nb) Factori pgmpy:", len(model.get_factors()))

    # c) Compute Z and verify P sums to 1.
    rows = enumerate_weights(config)
    z = sum(score for _, score in rows)
    assert np.isclose(sum(score / z for _, score in rows), 1.0)
    print(f"\nc) Configuratii: {len(rows)}; Z = {z:.9f}")

    # d) Infer each unknown station given S1=Soare, S5=Nori.
    evidence: dict[int, int] = {0: 0, 4: 2}
    compatible = filter_evidence(rows, evidence, config)
    z_evidence = sum(score for _, score in compatible)
    print(f"\nd) Configuratii compatibile: {len(compatible)}; Z_evidence={z_evidence:.9f}")
    marginal_map: dict[int, int] = {}
    for i in range(config.station_count):
        if i in evidence:
            continue
        probs = marginal(compatible, i, z_evidence)
        marginal_map[i] = int(np.argmax(probs))
        formatted = {label: round(float(p), 6) for label, p in zip(STATES, probs)}
        print(f"   S{i+1}: {formatted}; MAP marginal = {STATES[marginal_map[i]]}")

    # e) Determine joint MAP and compare with independent marginal choices.
    best, probability = global_map(compatible, z_evidence)
    print("\ne) MAP global:", {name: STATES[state] for name, state in zip(config.names, best)})
    print(f"   P(MAP global | observatii) = {probability:.6f}")
    print("   Coincide cu maximele marginale:",
          all(best[i] == value for i, value in marginal_map.items()))


if __name__ == "__main__":
    main()
