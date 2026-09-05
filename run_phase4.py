from __future__ import annotations

from config import load_config

from state_environment import (
    evaluate_all_routing_actions,
    export_state_evaluations,
    print_state_summary,
)

from topology import load_topology_snapshot


def main() -> None:
    """
    Run Phase 4 state and reward construction.
    """

    try:
        config = load_config()

        topology = load_topology_snapshot(
            config
        )

        evaluations = evaluate_all_routing_actions(
            topology=topology,
            config=config,
        )

        export_state_evaluations(
            evaluations=evaluations,
            config=config,
        )

        print_state_summary(
            evaluations=evaluations,
            config=config,
        )

    except (
        ValueError,
        FileNotFoundError,
    ) as error:

        print("=" * 68)
        print("PHASE 4 ERROR")
        print("=" * 68)
        print(error)

        raise SystemExit(1) from error


if __name__ == "__main__":
    main()