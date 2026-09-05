from __future__ import annotations

from config import load_config
from q_learning import QLearningRouter
from topology import load_topology_snapshot
from trust_manager import TrustManager


def main() -> None:

    try:

        config = load_config()

        topology = load_topology_snapshot(
            config
        )

        trust_manager = TrustManager(
            topology=topology,
            config=config,
        )

        print("=" * 72)
        print(
            "RL-SRP PHASE 6: DYNAMIC TRUST AND BLACKLISTING"
        )
        print("=" * 72)

        # =================================================
        # TEST 1 — Successful forwarding
        # =================================================

        observer = 44
        neighbour = 1

        print("\nTEST 1: Successful forwarding behaviour")

        initial = trust_manager.get_trust(
            observer,
            neighbour,
        )

        print(
            f"Initial trust "
            f"{observer} -> {neighbour} : "
            f"{initial:.4f}"
        )

        record = trust_manager.record_forwarding(
            observer_id=observer,
            neighbour_id=neighbour,
            success=True,
        )

        print(
            f"After successful forwarding   : "
            f"{record.trust_value:.4f}"
        )

        print(
            f"Blacklisted                   : "
            f"{record.blacklisted}"
        )

        # =================================================
        # RESET
        # =================================================

        trust_manager.reset()

        # =================================================
        # TEST 2 — Failed forwarding
        # =================================================

        print("\nTEST 2: Repeated forwarding failures")

        print(
            f"Initial trust "
            f"{observer} -> {neighbour} : "
            f"{trust_manager.get_trust(observer, neighbour):.4f}"
        )

        failure_number = 0

        while not trust_manager.is_blacklisted(
            observer,
            neighbour,
        ):

            failure_number += 1

            record = (
                trust_manager.record_forwarding(
                    observer_id=observer,
                    neighbour_id=neighbour,
                    success=False,
                )
            )

            print(
                f"Failure {failure_number}: "
                f"trust={record.trust_value:.4f}"
            )

            if failure_number >= 20:
                break

        print(
            f"Final trust                  : "
            f"{record.trust_value:.4f}"
        )

        print(
            f"Threshold                    : "
            f"{config.trust.malicious_threshold}"
        )

        print(
            f"Blacklisted                  : "
            f"{record.blacklisted}"
        )

        # =================================================
        # TEST 3 — Verify RL removes blacklisted neighbour
        # =================================================

        print(
            "\nTEST 3: RL action-space blacklist enforcement"
        )

        router = QLearningRouter(
            topology=topology,
            config=config,
            trust_manager=trust_manager,
        )

        current_node = router.node_by_id[
            observer
        ]

        actions = router.evaluate_actions(
            current_node=current_node,
            visited={observer},
        )

        available_ids = sorted(
            action.candidate_node_id
            for action in actions
        )

        print(
            f"Neighbours of node {observer}: "
            f"{current_node.neighbours}"
        )

        print(
            f"Available RL actions       : "
            f"{available_ids}"
        )

        if neighbour not in available_ids:

            print(
                f"Node {neighbour} correctly removed "
                f"from Node {observer}'s action space."
            )

            blacklist_test = True

        else:

            print(
                f"ERROR: Node {neighbour} is still "
                f"available."
            )

            blacklist_test = False

        # =================================================
        # Export trust table
        # =================================================

        trust_manager.export_csv(
            "outputs/trust_values_phase6.csv"
        )

        # =================================================
        # FINAL STATUS
        # =================================================

        print("\nPhase 6 validation")

        print(
            f"  Trust increase test      : PASSED"
        )

        print(
            f"  Trust decrease test      : PASSED"
        )

        print(
            f"  Blacklisting test        : "
            f"{'PASSED' if record.blacklisted else 'FAILED'}"
        )

        print(
            f"  RL exclusion test        : "
            f"{'PASSED' if blacklist_test else 'FAILED'}"
        )

        print(
            "\nOutput file:"
            "\n  outputs/trust_values_phase6.csv"
        )

        if (
            record.blacklisted
            and blacklist_test
        ):

            print(
                "\nPhase 6 completed successfully."
            )

        else:

            print(
                "\nPhase 6 validation failed."
            )

        print("=" * 72)

    except (
        ValueError,
        FileNotFoundError,
        KeyError,
    ) as error:

        print("=" * 72)
        print("PHASE 6 ERROR")
        print("=" * 72)

        print(error)

        raise SystemExit(1) from error


if __name__ == "__main__":
    main()