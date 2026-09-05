from __future__ import annotations

import csv

from pathlib import Path

from config import ProjectConfig
from q_learning import (
    QLearningRouter,
    RLRoutingResult,
    TrainingEpisodeResult,
)


def export_training_results(
    config: ProjectConfig,
    results: list[TrainingEpisodeResult],
) -> Path:

    output_directory = Path(
        config.output.directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        output_directory
        / "q_learning_training.csv"
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "episode",
                "source_id",
                "delivered",
                "hops",
                "total_reward",
                "epsilon",
            ]
        )

        for result in results:

            writer.writerow(
                [
                    result.episode,
                    result.source_id,
                    result.delivered,
                    result.hops,
                    f"{result.total_reward:.6f}",
                    f"{result.epsilon:.6f}",
                ]
            )

    return path


def export_q_table(
    config: ProjectConfig,
    router: QLearningRouter,
) -> Path:

    output_directory = Path(
        config.output.directory
    )

    path = (
        output_directory
        / "q_table.csv"
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "current_node_id",
                "energy_bin",
                "link_quality_bin",
                "trust_bin",
                "distance_bin",
                "action_neighbor_id",
                "q_value",
            ]
        )

        for key, q_value in sorted(
            router.q_table.items()
        ):

            (
                current_node_id,
                state_tuple,
                action_neighbor_id,
            ) = key

            (
                energy_bin,
                link_quality_bin,
                trust_bin,
                distance_bin,
            ) = state_tuple

            writer.writerow(
                [
                    current_node_id,
                    energy_bin,
                    link_quality_bin,
                    trust_bin,
                    distance_bin,
                    action_neighbor_id,
                    f"{q_value:.8f}",
                ]
            )

    return path


def export_rl_routing_results(
    config: ProjectConfig,
    results: list[RLRoutingResult],
) -> Path:

    output_directory = Path(
        config.output.directory
    )

    path = (
        output_directory
        / "rl_packet_routing_results.csv"
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "packet_id",
                "source_id",
                "delivered",
                "hop_count",
                "route",
                "total_reward",
                "reason",
            ]
        )

        for result in results:

            writer.writerow(
                [
                    result.packet_id,
                    result.source_id,
                    result.delivered,
                    result.hop_count,
                    " -> ".join(
                        map(str, result.route)
                    ),
                    f"{result.total_reward:.6f}",
                    result.reason,
                ]
            )

    return path