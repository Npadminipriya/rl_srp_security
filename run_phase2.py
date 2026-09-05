from __future__ import annotations

from config import load_config

from topology import (
    TopologyGenerationError,
    export_topology,
    generate_valid_topology,
    print_topology_summary,
    save_topology_snapshot,
)


def main() -> None:
    """
    Run the complete Phase 2 process.
    """

    try:
        # Load and validate Phase 1 settings
        config = load_config()

        # Generate a connected random topology
        result = generate_valid_topology(
            config
        )

        # Save topology files
        export_topology(
            config=config,
            result=result,
        )

        # Save a snapshot of the generated topology
        save_topology_snapshot(
            config=config,
            result=result,
        )

        # Display summary
        print_topology_summary(
            config=config,
            result=result,
        )

    except (
        ValueError,
        FileNotFoundError,
        TopologyGenerationError,
    ) as error:

        print("=" * 62)
        print("PHASE 2 ERROR")
        print("=" * 62)
        print(error)

        raise SystemExit(1) from error


if __name__ == "__main__":
    main()