from __future__ import annotations

from config import load_config

from phase5_results import (
    export_q_table,
    export_rl_routing_results,
    export_training_results,
)

from q_learning import QLearningRouter
from topology import load_topology_snapshot


def main() -> None:

    try:

        config = load_config()

        topology = load_topology_snapshot(
            config
        )

        print("=" * 68)
        print(
            "RL-SRP PHASE 5: Q-LEARNING ROUTING"
        )
        print("=" * 68)

        # ---------------------------------
        # Training
        # ---------------------------------

        router = QLearningRouter(
            topology=topology,
            config=config,
        )

        print(
            f"Training episodes           : "
            f"{config.q_learning.training_episodes}"
        )

        print(
            f"Initial epsilon             : "
            f"{config.q_learning.epsilon_start}"
        )

        print(
            f"Minimum epsilon             : "
            f"{config.q_learning.epsilon_min}"
        )

        print("\nTraining Q-learning agent...")

        training_results = router.train()

        export_training_results(
            config=config,
            results=training_results,
        )

        export_q_table(
            config=config,
            router=router,
        )

        # ---------------------------------
        # Training statistics
        # ---------------------------------

        total_training = len(
            training_results
        )

        successful_training = sum(
            result.delivered
            for result in training_results
        )

        training_success_rate = (
            successful_training
            / total_training
            * 100
        )

        last_window = (
            training_results[-500:]
        )

        recent_successful = sum(
            result.delivered
            for result in last_window
        )

        recent_success_rate = (
            recent_successful
            / len(last_window)
            * 100
        )

        # ---------------------------------
        # Testing
        # ---------------------------------

        print("\nTesting learned policy...")

        routing_results = []

        packet_id = 1

        for node in topology.nodes:

            if node.node_type != "sensor":
                continue

            result = router.route_packet(
                packet_id=packet_id,
                source_id=node.node_id,
            )

            routing_results.append(
                result
            )

            packet_id += 1

        export_rl_routing_results(
            config=config,
            results=routing_results,
        )

        delivered = sum(
            result.delivered
            for result in routing_results
        )

        dropped = (
            len(routing_results)
            - delivered
        )

        pdr = (
            delivered
            / len(routing_results)
            * 100
        )

        delivered_hops = [
            result.hop_count
            for result in routing_results
            if result.delivered
        ]

        average_hops = (
            sum(delivered_hops)
            / len(delivered_hops)
            if delivered_hops
            else 0.0
        )

        # ---------------------------------
        # Output
        # ---------------------------------

        print("\nTraining results")

        print(
            f"  Overall success rate      : "
            f"{training_success_rate:.2f}%"
        )

        print(
            f"  Last 500 success rate     : "
            f"{recent_success_rate:.2f}%"
        )

        print(
            f"  Final epsilon             : "
            f"{router.epsilon:.6f}"
        )

        print(
            f"  Learned Q-table entries   : "
            f"{len(router.q_table)}"
        )

        print("\nRL routing test")

        print(
            f"  Packets generated         : "
            f"{len(routing_results)}"
        )

        print(
            f"  Packets delivered         : "
            f"{delivered}"
        )

        print(
            f"  Packets dropped           : "
            f"{dropped}"
        )

        print(
            f"  Packet delivery ratio     : "
            f"{pdr:.2f}%"
        )

        print(
            f"  Average hop count         : "
            f"{average_hops:.2f}"
        )

        print(
            "\nPhase 3 greedy PDR          : "
            "88.00%"
        )

        difference = (
            pdr - 88.0
        )

        print(
            f"RL improvement              : "
            f"{difference:+.2f} percentage points"
        )

        print("\nSample learned routes")

        for result in routing_results[:10]:

            route = " -> ".join(
                map(str, result.route)
            )

            status = (
                "DELIVERED"
                if result.delivered
                else "DROPPED"
            )

            print(
                f"  Packet {result.packet_id:02d}: "
                f"{route} [{status}]"
            )

        print("\nOutput files")

        print(
            "  outputs/q_learning_training.csv"
        )

        print(
            "  outputs/q_table.csv"
        )

        print(
            "  outputs/rl_packet_routing_results.csv"
        )

        print(
            "\nPhase 5 completed successfully."
        )

        print("=" * 68)

    except (
        ValueError,
        FileNotFoundError,
    ) as error:

        print("=" * 68)
        print("PHASE 5 ERROR")
        print("=" * 68)

        print(error)

        raise SystemExit(1) from error


if __name__ == "__main__":
    main()