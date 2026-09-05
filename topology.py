from __future__ import annotations

import csv
import json
import random

from collections import deque
from dataclasses import dataclass
from pathlib import Path

from config import ProjectConfig
from node import Node


@dataclass
class Link:
    """
    Represents one communication connection
    between two neighbouring nodes.
    """

    source_id: int
    target_id: int
    distance_m: float


@dataclass
class TopologyResult:
    """
    Stores the successfully generated topology.
    """

    nodes: list[Node]
    links: list[Link]
    generation_seed: int
    generation_attempt: int


class TopologyGenerationError(RuntimeError):
    """
    Raised when no valid topology can be generated.
    """

    pass


def create_nodes(
    config: ProjectConfig,
    rng: random.Random,
) -> list[Node]:
    """
    Create one sink and randomly deploy all sensor nodes.
    """

    sink = Node(
        node_id=0,
        node_type="sink",
        x=config.sink.x_m,
        y=config.sink.y_m,
        z=config.sink.z_m,
        initial_energy_j=None,
        residual_energy_j=None,
    )

    nodes = [sink]

    for node_id in range(
        1,
        config.topology.sensor_nodes + 1,
    ):
        sensor = Node(
            node_id=node_id,
            node_type="sensor",

            x=rng.uniform(
                0.0,
                config.topology.area_x_m,
            ),

            y=rng.uniform(
                0.0,
                config.topology.area_y_m,
            ),

            z=rng.uniform(
                config.topology.minimum_sensor_depth_m,
                config.topology.depth_m,
            ),

            initial_energy_j=(
                config.energy.initial_energy_j
            ),

            residual_energy_j=(
                config.energy.initial_energy_j
            ),
        )

        nodes.append(sensor)

    return nodes


def build_links(
    nodes: list[Node],
    communication_range_m: float,
) -> list[Link]:
    """
    Find all node pairs that are within
    communication range.
    """

    links: list[Link] = []

    for index, source in enumerate(nodes):

        for target in nodes[index + 1:]:

            distance = source.distance_to(target)

            if distance <= communication_range_m:

                source.neighbours.append(
                    target.node_id
                )

                target.neighbours.append(
                    source.node_id
                )

                link = Link(
                    source_id=source.node_id,
                    target_id=target.node_id,
                    distance_m=distance,
                )

                links.append(link)

    for node in nodes:
        node.neighbours.sort()

    return links


def minimum_sensor_degree_met(
    nodes: list[Node],
    minimum_neighbors: int,
) -> bool:
    """
    Check whether every sensor node has at least
    the required minimum number of neighbours.
    """

    for node in nodes:

        if node.node_type != "sensor":
            continue

        if len(node.neighbours) < minimum_neighbors:
            return False

    return True


def all_sensors_reach_sink(
    nodes: list[Node],
) -> bool:
    """
    Use Breadth-First Search to check whether
    every sensor has a multi-hop path to sink 0.
    """

    node_by_id = {
        node.node_id: node
        for node in nodes
    }

    visited = {0}
    queue: deque[int] = deque([0])

    while queue:

        current_id = queue.popleft()
        current_node = node_by_id[current_id]

        for neighbour_id in current_node.neighbours:

            if neighbour_id not in visited:

                visited.add(neighbour_id)
                queue.append(neighbour_id)

    sensor_ids = {
        node.node_id
        for node in nodes
        if node.node_type == "sensor"
    }

    return sensor_ids.issubset(visited)


def generate_valid_topology(
    config: ProjectConfig,
) -> TopologyResult:
    """
    Repeatedly generate random deployments until:

    1. Every sensor has the required minimum neighbours.
    2. Every sensor can reach the surface sink.
    """

    base_seed = config.simulation.random_seed

    maximum_attempts = (
        config.topology.maximum_generation_attempts
    )

    for attempt in range(
        1,
        maximum_attempts + 1,
    ):

        attempt_seed = (
            base_seed + attempt - 1
        )

        rng = random.Random(attempt_seed)

        nodes = create_nodes(
            config=config,
            rng=rng,
        )

        links = build_links(
            nodes=nodes,
            communication_range_m=(
                config.topology.communication_range_m
            ),
        )

        minimum_degree_valid = (
            minimum_sensor_degree_met(
                nodes=nodes,
                minimum_neighbors=(
                    config.topology.minimum_neighbors
                ),
            )
        )

        if not minimum_degree_valid:
            continue

        sink_reachable = all_sensors_reach_sink(
            nodes
        )

        if not sink_reachable:
            continue

        return TopologyResult(
            nodes=nodes,
            links=links,
            generation_seed=attempt_seed,
            generation_attempt=attempt,
        )

    raise TopologyGenerationError(
        "Unable to generate a valid connected topology "
        f"after {maximum_attempts} attempts. "
        "Try increasing communication_range_m, "
        "reducing minimum_neighbors, or reducing "
        "the deployment area."
    )


def export_topology(
    config: ProjectConfig,
    result: TopologyResult,
) -> Path:
    """
    Export generated topology data into CSV
    and JSON files.
    """

    output_directory = Path(
        config.output.directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    sink = next(
        node
        for node in result.nodes
        if node.node_type == "sink"
    )

    # -----------------------------------
    # Export nodes.csv
    # -----------------------------------

    nodes_path = (
        output_directory / "nodes.csv"
    )

    with nodes_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "node_id",
                "node_type",
                "x_m",
                "y_m",
                "z_m",
                "initial_energy_j",
                "residual_energy_j",
                "distance_to_sink_m",
                "neighbor_count",
            ]
        )

        for node in result.nodes:

            initial_energy = (
                "INF"
                if node.initial_energy_j is None
                else f"{node.initial_energy_j:.3f}"
            )

            residual_energy = (
                "INF"
                if node.residual_energy_j is None
                else f"{node.residual_energy_j:.3f}"
            )

            writer.writerow(
                [
                    node.node_id,
                    node.node_type,
                    f"{node.x:.3f}",
                    f"{node.y:.3f}",
                    f"{node.z:.3f}",
                    initial_energy,
                    residual_energy,
                    f"{node.distance_to_sink(sink):.3f}",
                    len(node.neighbours),
                ]
            )

    # -----------------------------------
    # Export links.csv
    # -----------------------------------

    links_path = (
        output_directory / "links.csv"
    )

    with links_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "source_id",
                "target_id",
                "distance_m",
            ]
        )

        for link in result.links:

            writer.writerow(
                [
                    link.source_id,
                    link.target_id,
                    f"{link.distance_m:.3f}",
                ]
            )

    # -----------------------------------
    # Export neighbors.csv
    # -----------------------------------

    neighbors_path = (
        output_directory / "neighbors.csv"
    )

    with neighbors_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "node_id",
                "neighbor_id",
                "initial_trust",
            ]
        )

        for node in result.nodes:

            for neighbour_id in node.neighbours:

                writer.writerow(
                    [
                        node.node_id,
                        neighbour_id,
                        (
                            f"{config.trust.initial_value:.3f}"
                        ),
                    ]
                )

    # -----------------------------------
    # Export topology_summary.json
    # -----------------------------------

    sensor_nodes = [
        node
        for node in result.nodes
        if node.node_type == "sensor"
    ]

    neighbour_counts = [
        len(node.neighbours)
        for node in sensor_nodes
    ]

    summary = {
        "generation_seed": (
            result.generation_seed
        ),

        "generation_attempt": (
            result.generation_attempt
        ),

        "sensor_nodes": (
            config.topology.sensor_nodes
        ),

        "sink_nodes": (
            config.topology.sink_nodes
        ),

        "total_devices": (
            len(result.nodes)
        ),

        "undirected_links": (
            len(result.links)
        ),

        "communication_range_m": (
            config.topology.communication_range_m
        ),

        "minimum_neighbors_required": (
            config.topology.minimum_neighbors
        ),

        "minimum_sensor_neighbors_observed": (
            min(neighbour_counts)
        ),

        "maximum_sensor_neighbors_observed": (
            max(neighbour_counts)
        ),

        "average_sensor_neighbors": round(
            sum(neighbour_counts)
            / len(neighbour_counts),
            3,
        ),

        "all_sensors_reach_sink": True,
    }

    summary_path = (
        output_directory
        / "topology_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    return output_directory


def print_topology_summary(
    config: ProjectConfig,
    result: TopologyResult,
) -> None:
    """
    Print a readable Phase 2 summary.
    """

    sensor_nodes = [
        node
        for node in result.nodes
        if node.node_type == "sensor"
    ]

    neighbour_counts = [
        len(node.neighbours)
        for node in sensor_nodes
    ]

    average_neighbors = (
        sum(neighbour_counts)
        / len(neighbour_counts)
    )

    print("=" * 62)
    print(
        "RL-SRP PHASE 2: NETWORK ENVIRONMENT"
    )
    print("=" * 62)

    print(
        "Topology generation attempt : "
        f"{result.generation_attempt}"
    )

    print(
        "Topology random seed        : "
        f"{result.generation_seed}"
    )

    print(
        "Sensor nodes                : "
        f"{len(sensor_nodes)}"
    )

    print(
        "Surface sink nodes          : "
        f"{config.topology.sink_nodes}"
    )

    print(
        "Total devices               : "
        f"{len(result.nodes)}"
    )

    print(
        "Undirected communication links: "
        f"{len(result.links)}"
    )

    print(
        "Deployment region           : "
        f"{config.topology.area_x_m} × "
        f"{config.topology.area_y_m} × "
        f"{config.topology.depth_m} m"
    )

    print(
        "Communication range         : "
        f"{config.topology.communication_range_m} m"
    )

    print(
        "Minimum sensor neighbours   : "
        f"{min(neighbour_counts)}"
    )

    print(
        "Maximum sensor neighbours   : "
        f"{max(neighbour_counts)}"
    )

    print(
        "Average sensor neighbours   : "
        f"{average_neighbors:.2f}"
    )

    print(
        "All sensors can reach sink  : True"
    )

    print(
        "Initial neighbour trust     : "
        f"{config.trust.initial_value}"
    )

    print("\nOutput files")

    print(
        f"  {config.output.directory}/nodes.csv"
    )

    print(
        f"  {config.output.directory}/links.csv"
    )

    print(
        f"  {config.output.directory}/neighbors.csv"
    )

    print(
        f"  {config.output.directory}/"
        "topology_summary.json"
    )

    print(
        "\nPhase 2 completed successfully."
    )

    print("=" * 62)
def save_topology_snapshot(
    config: ProjectConfig,
    result: TopologyResult,
) -> None:
    """
    Save the complete generated topology so it can
    be reused by later phases.
    """

    output_directory = Path(
        config.output.directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot = {
        "generation_seed": result.generation_seed,
        "generation_attempt": result.generation_attempt,
        "nodes": [],
        "links": [],
    }

    for node in result.nodes:

        snapshot["nodes"].append(
            {
                "node_id": node.node_id,
                "node_type": node.node_type,
                "x": node.x,
                "y": node.y,
                "z": node.z,
                "initial_energy_j": node.initial_energy_j,
                "residual_energy_j": node.residual_energy_j,
                "neighbours": node.neighbours,
            }
        )

    for link in result.links:

        snapshot["links"].append(
            {
                "source_id": link.source_id,
                "target_id": link.target_id,
                "distance_m": link.distance_m,
            }
        )

    snapshot_path = (
        output_directory
        / "topology_snapshot.json"
    )

    snapshot_path.write_text(
        json.dumps(
            snapshot,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_topology_snapshot(
    config: ProjectConfig,
) -> TopologyResult:
    """
    Load the topology generated during Phase 2.
    """

    snapshot_path = (
        Path(config.output.directory)
        / "topology_snapshot.json"
    )

    if not snapshot_path.exists():
        raise FileNotFoundError(
            "topology_snapshot.json was not found. "
            "Run Phase 2 again before Phase 3."
        )

    raw = json.loads(
        snapshot_path.read_text(
            encoding="utf-8"
        )
    )

    nodes: list[Node] = []

    for node_data in raw["nodes"]:

        node = Node(
            node_id=node_data["node_id"],
            node_type=node_data["node_type"],
            x=node_data["x"],
            y=node_data["y"],
            z=node_data["z"],
            initial_energy_j=(
                node_data["initial_energy_j"]
            ),
            residual_energy_j=(
                node_data["residual_energy_j"]
            ),
            neighbours=(
                node_data["neighbours"]
            ),
        )

        nodes.append(node)

    links = [
        Link(
            source_id=link_data["source_id"],
            target_id=link_data["target_id"],
            distance_m=link_data["distance_m"],
        )
        for link_data in raw["links"]
    ]

    return TopologyResult(
        nodes=nodes,
        links=links,
        generation_seed=raw["generation_seed"],
        generation_attempt=(
            raw["generation_attempt"]
        ),
    )