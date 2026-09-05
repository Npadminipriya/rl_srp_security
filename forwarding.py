from __future__ import annotations

import csv

from dataclasses import dataclass
from pathlib import Path

from config import ProjectConfig
from node import Node
from packet import Packet
from topology import TopologyResult


@dataclass
class ForwardingResult:
    """
    Stores the result of forwarding one packet.
    """

    packet: Packet
    success: bool
    reason: str


def choose_next_hop(
    current_node: Node,
    sink: Node,
    node_by_id: dict[int, Node],
    visited: set[int],
) -> Node | None:
    """
    Select the neighbouring node that is closest
    to the sink.

    Only neighbours that:
    1. Have not already been visited.
    2. Are closer to the sink than the current node.
    are considered.
    """

    current_distance = current_node.distance_to_sink(sink)

    candidates: list[Node] = []

    for neighbour_id in current_node.neighbours:

        if neighbour_id in visited:
            continue

        neighbour = node_by_id[neighbour_id]

        neighbour_distance = neighbour.distance_to_sink(sink)

        if neighbour_distance < current_distance:
            candidates.append(neighbour)

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda node: node.distance_to_sink(sink),
    )


def forward_packet(
    packet: Packet,
    topology: TopologyResult,
    maximum_hops: int = 100,
) -> ForwardingResult:
    """
    Forward a packet from its source to the sink
    using greedy next-hop selection.
    """

    node_by_id = {
        node.node_id: node
        for node in topology.nodes
    }

    sink = node_by_id[packet.destination_id]

    current_node = node_by_id[packet.source_id]

    packet.add_hop(current_node.node_id)

    visited = {current_node.node_id}

    while current_node.node_id != sink.node_id:

        if packet.hop_count >= maximum_hops:
            packet.dropped = True
            packet.drop_reason = "maximum hop limit reached"

            return ForwardingResult(
                packet=packet,
                success=False,
                reason=packet.drop_reason,
            )

        next_hop = choose_next_hop(
            current_node=current_node,
            sink=sink,
            node_by_id=node_by_id,
            visited=visited,
        )

        if next_hop is None:
            packet.dropped = True
            packet.drop_reason = (
                "no neighbour closer to the sink"
            )

            return ForwardingResult(
                packet=packet,
                success=False,
                reason=packet.drop_reason,
            )

        packet.add_hop(next_hop.node_id)

        visited.add(next_hop.node_id)
        current_node = next_hop

    packet.delivered = True

    return ForwardingResult(
        packet=packet,
        success=True,
        reason="packet delivered successfully",
    )


def create_packets(
    topology: TopologyResult,
    packets_per_sensor: int = 1,
) -> list[Packet]:
    """
    Generate packets from every sensor node.

    The destination is always sink node 0.
    """

    packets: list[Packet] = []

    packet_id = 1

    for node in topology.nodes:

        if node.node_type != "sensor":
            continue

        for packet_number in range(
            packets_per_sensor
        ):
            packet = Packet(
                packet_id=packet_id,
                source_id=node.node_id,
                destination_id=0,
                payload=(
                    f"Sensor data from node {node.node_id}, "
                    f"packet {packet_number + 1}"
                ),
                current_node_id=node.node_id,
            )

            packets.append(packet)
            packet_id += 1

    return packets


def export_packet_results(
    config: ProjectConfig,
    results: list[ForwardingResult],
) -> Path:
    """
    Save packet-forwarding results into CSV files.
    """

    output_directory = Path(
        config.output.directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    packet_results_path = (
        output_directory
        / "packet_forwarding_results.csv"
    )

    with packet_results_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "packet_id",
                "source_id",
                "destination_id",
                "delivered",
                "dropped",
                "hop_count",
                "route",
                "result_reason",
            ]
        )

        for result in results:

            packet = result.packet

            writer.writerow(
                [
                    packet.packet_id,
                    packet.source_id,
                    packet.destination_id,
                    packet.delivered,
                    packet.dropped,
                    packet.hop_count,
                    " -> ".join(
                        map(str, packet.route)
                    ),
                    result.reason,
                ]
            )

    return packet_results_path


def print_forwarding_summary(
    results: list[ForwardingResult],
) -> None:
    """
    Print overall packet-forwarding statistics.
    """

    total_packets = len(results)

    delivered_packets = sum(
        1
        for result in results
        if result.success
    )

    dropped_packets = (
        total_packets - delivered_packets
    )

    packet_delivery_ratio = (
        delivered_packets / total_packets
        if total_packets > 0
        else 0.0
    )

    delivered_hops = [
        result.packet.hop_count
        for result in results
        if result.success
    ]

    average_hops = (
        sum(delivered_hops) / len(delivered_hops)
        if delivered_hops
        else 0.0
    )

    print("=" * 62)
    print("RL-SRP PHASE 3: BASIC PACKET FORWARDING")
    print("=" * 62)

    print(f"Total packets generated    : {total_packets}")
    print(f"Packets delivered          : {delivered_packets}")
    print(f"Packets dropped            : {dropped_packets}")

    print(
        "Packet delivery ratio      : "
        f"{packet_delivery_ratio * 100:.2f}%"
    )

    print(
        "Average hop count          : "
        f"{average_hops:.2f}"
    )

    print("\nSample routes")

    for result in results[:10]:

        packet = result.packet

        route_text = " -> ".join(
            map(str, packet.route)
        )

        status = (
            "DELIVERED"
            if packet.delivered
            else "DROPPED"
        )

        print(
            f"  Packet {packet.packet_id:02d}: "
            f"{route_text} [{status}]"
        )

    print(
        "\nOutput file:"
        "\n  outputs/packet_forwarding_results.csv"
    )

    print("\nPhase 3 completed.")
    print("=" * 62)