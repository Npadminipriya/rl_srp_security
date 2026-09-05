from __future__ import annotations

from config import load_config

from forwarding import (
    create_packets,
    export_packet_results,
    forward_packet,
    print_forwarding_summary,
)

from topology import load_topology_snapshot


def main() -> None:
    """
    Run Phase 3 basic packet forwarding.
    """

    try:
        config = load_config()

        topology = load_topology_snapshot(
            config
        )

        packets = create_packets(
            topology=topology,
            packets_per_sensor=1,
        )

        results = []

        for packet in packets:

            result = forward_packet(
                packet=packet,
                topology=topology,
                maximum_hops=100,
            )

            results.append(result)

        export_packet_results(
            config=config,
            results=results,
        )

        print_forwarding_summary(
            results
        )

    except (
        ValueError,
        FileNotFoundError,
    ) as error:

        print("=" * 62)
        print("PHASE 3 ERROR")
        print("=" * 62)
        print(error)

        raise SystemExit(1) from error


if __name__ == "__main__":
    main()