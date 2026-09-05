from __future__ import annotations

import csv

from pathlib import Path

from config import ProjectConfig
from rl_state import CandidateEvaluation, evaluate_candidate
from topology import TopologyResult


def evaluate_all_routing_actions(
    topology: TopologyResult,
    config: ProjectConfig,
) -> list[CandidateEvaluation]:
    """
    Evaluate every sensor-to-neighbour routing action
    available in the generated topology.
    """

    node_by_id = {
        node.node_id: node
        for node in topology.nodes
    }

    sink = node_by_id[0]

    evaluations: list[CandidateEvaluation] = []

    for current_node in topology.nodes:

        if current_node.node_type != "sensor":
            continue

        for neighbour_id in current_node.neighbours:

            candidate = node_by_id[neighbour_id]

            trust_value = (
                config.trust.initial_value
            )

            evaluation = evaluate_candidate(
                current_node=current_node,
                candidate=candidate,
                sink=sink,
                trust_value=trust_value,
                config=config,
            )

            evaluations.append(evaluation)

    return evaluations


def export_state_evaluations(
    evaluations: list[CandidateEvaluation],
    config: ProjectConfig,
) -> Path:
    """
    Export all RL state vectors and reward components.
    """

    output_directory = Path(
        config.output.directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / "rl_state_candidates.csv"
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "current_node_id",
                "candidate_node_id",
                "link_distance_m",

                "residual_energy",
                "link_quality",
                "trust_value",
                "normalized_distance_to_sink",

                "energy_bin",
                "link_quality_bin",
                "trust_bin",
                "distance_bin",
                "state_tuple",

                "distance_progress",
                "estimated_energy_cost",
                "estimated_delay",
                "estimated_packet_loss_rate",

                "preview_reward",
            ]
        )

        for evaluation in evaluations:

            state = evaluation.state
            discrete = evaluation.discrete_state
            reward = evaluation.reward

            writer.writerow(
                [
                    evaluation.current_node_id,
                    evaluation.candidate_node_id,
                    f"{evaluation.link_distance_m:.6f}",

                    f"{state.residual_energy:.6f}",
                    f"{state.link_quality:.6f}",
                    f"{state.trust_value:.6f}",
                    f"{state.distance_to_sink:.6f}",

                    discrete.energy_bin,
                    discrete.link_quality_bin,
                    discrete.trust_bin,
                    discrete.distance_bin,
                    str(discrete.as_tuple()),

                    f"{reward.distance_progress:.6f}",
                    f"{reward.energy_cost:.6f}",
                    f"{reward.delay:.6f}",
                    f"{reward.packet_loss_rate:.6f}",

                    f"{reward.total_reward:.6f}",
                ]
            )

    return output_path


def print_state_summary(
    evaluations: list[CandidateEvaluation],
    config: ProjectConfig,
) -> None:
    """
    Display Phase 4 evaluation statistics.
    """

    if not evaluations:
        print("No routing actions were available.")
        return

    rewards = [
        evaluation.reward.total_reward
        for evaluation in evaluations
    ]

    average_reward = (
        sum(rewards) / len(rewards)
    )

    positive_actions = sum(
        reward > 0
        for reward in rewards
    )

    negative_actions = sum(
        reward < 0
        for reward in rewards
    )

    neutral_actions = (
        len(rewards)
        - positive_actions
        - negative_actions
    )

    best_evaluation = max(
        evaluations,
        key=lambda item: item.reward.total_reward,
    )

    worst_evaluation = min(
        evaluations,
        key=lambda item: item.reward.total_reward,
    )

    print("=" * 68)
    print("RL-SRP PHASE 4: STATE AND REWARD ENVIRONMENT")
    print("=" * 68)

    print(
        f"Candidate routing actions     : "
        f"{len(evaluations)}"
    )

    print(
        f"Positive preview rewards      : "
        f"{positive_actions}"
    )

    print(
        f"Negative preview rewards      : "
        f"{negative_actions}"
    )

    print(
        f"Neutral preview rewards       : "
        f"{neutral_actions}"
    )

    print(
        f"Average preview reward        : "
        f"{average_reward:.6f}"
    )

    print(
        f"Highest preview reward        : "
        f"{max(rewards):.6f}"
    )

    print(
        f"Lowest preview reward         : "
        f"{min(rewards):.6f}"
    )

    print("\nBest initial action")

    print(
        f"  Current node                : "
        f"{best_evaluation.current_node_id}"
    )

    print(
        f"  Candidate neighbour         : "
        f"{best_evaluation.candidate_node_id}"
    )

    print(
        f"  State                       : "
        f"{best_evaluation.discrete_state.as_tuple()}"
    )

    print(
        f"  Reward                      : "
        f"{best_evaluation.reward.total_reward:.6f}"
    )

    print("\nWorst initial action")

    print(
        f"  Current node                : "
        f"{worst_evaluation.current_node_id}"
    )

    print(
        f"  Candidate neighbour         : "
        f"{worst_evaluation.candidate_node_id}"
    )

    print(
        f"  State                       : "
        f"{worst_evaluation.discrete_state.as_tuple()}"
    )

    print(
        f"  Reward                      : "
        f"{worst_evaluation.reward.total_reward:.6f}"
    )

    print("\nSample candidate evaluations")

    for evaluation in evaluations[:10]:

        print(
            f"  Node {evaluation.current_node_id:02d} "
            f"→ {evaluation.candidate_node_id:02d} | "
            f"state={evaluation.discrete_state.as_tuple()} | "
            f"reward="
            f"{evaluation.reward.total_reward:.4f}"
        )

    print(
        "\nOutput file:"
        "\n  outputs/rl_state_candidates.csv"
    )

    print("\nPhase 4 completed successfully.")
    print("=" * 68)