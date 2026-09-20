from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from config import load_config
from q_learning import QLearningRouter
from topology import load_topology_snapshot
from trust_manager import TrustManager
from replay_attack import ReplayAttack


# ============================================================
# EXPORT ATTACK RESULTS
# ============================================================

def export_attack_results(
    results: list[dict],
    path: str | Path,
) -> None:

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "packet_id",
                "source_id",
                "route",
                "hop_count",
                "rl_route_found",
                "packet_delivered",
                "dropped_by_replay",
                "malicious_node",
                "observer_node",
                "trust_before",
                "trust_after",
                "blacklisted",
                "result",
            ]
        )

        for result in results:

            writer.writerow(
                [
                    result["packet_id"],
                    result["source_id"],
                    "->".join(
                        map(
                            str,
                            result["route"],
                        )
                    ),
                    result["hop_count"],
                    result["rl_route_found"],
                    result["packet_delivered"],
                    result["dropped_by_replay"],
                    result["malicious_node"],
                    result["observer_node"],
                    (
                        f'{result["trust_before"]:.6f}'
                        if result["trust_before"] is not None
                        else ""
                    ),
                    (
                        f'{result["trust_after"]:.6f}'
                        if result["trust_after"] is not None
                        else ""
                    ),
                    result["blacklisted"],
                    result["result"],
                ]
            )


# ============================================================
# FIND A FREQUENTLY USED RELAY
# ============================================================

def find_attack_target(
    router: QLearningRouter,
    sensor_ids: list[int],
) -> int:

    relay_frequency = Counter()

    packet_id = 1

    for source_id in sensor_ids:

        routing_result = router.route_packet(
            packet_id=packet_id,
            source_id=source_id,
        )

        if routing_result.delivered:

            # Exclude source and sink.
            for node_id in routing_result.route[1:-1]:

                relay_frequency[node_id] += 1

        packet_id += 1

    if not relay_frequency:

        raise RuntimeError(
            "Could not find a relay node used by "
            "the learned routing policy."
        )

    return relay_frequency.most_common(1)[0][0]


# ============================================================
# MAIN
# ============================================================

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

        router = QLearningRouter(
            topology=topology,
            config=config,
            trust_manager=trust_manager,
        )

        print("=" * 72)
        print(
            "RL-SRP PHASE 10: REPLAY ATTACK"
        )
        print("=" * 72)

        # ====================================================
        # SENSOR LIST
        # ====================================================

        sensor_ids = [
            node.node_id
            for node in topology.nodes
            if node.node_type == "sensor"
        ]

        # ====================================================
        # TRAIN RL ROUTER
        # ====================================================

        print("\nTraining Q-learning agent...")

        training_results = router.train()

        successful_training = sum(
            result.delivered
            for result in training_results
        )

        training_rate = (
            successful_training
            / len(training_results)
            * 100
            if training_results
            else 0.0
        )

        print(
            f"Training episodes       : "
            f"{len(training_results)}"
        )

        print(
            f"Training success rate    : "
            f"{training_rate:.2f}%"
        )

        print(
            f"Final epsilon            : "
            f"{router.epsilon:.6f}"
        )

        print(
            f"Q-table entries          : "
            f"{len(router.q_table)}"
        )

        # ====================================================
        # FIND ATTACK TARGET
        # ====================================================

        malicious_node = find_attack_target(
            router=router,
            sensor_ids=sensor_ids,
        )

        replay_probability = 0.30

        print(
            "\nReplay attack configuration"
        )

        print(
            f"Malicious node           : "
            f"{malicious_node}"
        )

        print(
            "Attack behaviour         : "
            "Replay old packets"
        )

        print(
            f"Replay probability       : "
            f"{replay_probability:.0%}"
        )

        print(
            "Detection mechanism      : "
            "Dynamic trust + blacklist"
        )

        # ====================================================
        # ATTACK CREATION
        # ====================================================

        attack = ReplayAttack(
            malicious_node_id=malicious_node,
            trust_manager=trust_manager,
            replay_probability=replay_probability,
        )


        # ====================================================
        # ATTACK TEST
        # ====================================================

        print(
            "\nTesting packet forwarding "
            "under replay attack..."
        )

        results: list[dict] = []

        packet_id = 1

        # Number of packets generated by each sensor.
        packets_per_sensor = 5

        for source_id in sensor_ids:

            for packet_number in range(
                packets_per_sensor
            ):

                routing_result = router.route_packet(
                    packet_id=packet_id,
                    source_id=source_id,
                )

                route = routing_result.route

                dropped = False
                observer_id = None
                trust_before = None
                trust_after = None
                blacklisted = False

                # ------------------------------------------------
                # Examine each hop.
                # ------------------------------------------------

                if routing_result.delivered:

                    for index in range(
                        len(route) - 1
                    ):

                        current_id = route[index]

                        next_id = route[index + 1]

                        # Simulate forwarding to next node.
                        forwarding_success = (
                            attack.process_hop(
                                packet_id=packet_id,
                                observer_id=current_id,
                                next_node_id=next_id,
                            )
                        )   
                        if not forwarding_success:

                            dropped = True
                            observer_id = current_id

                            record = (
                                trust_manager.records[
                                    (
                                        current_id,
                                        next_id,
                                    )
                                ]
                            )

                            trust_before = (
                                record.trust_value
                            )

                            # The trust value is already updated
                            # by process_hop().
                            trust_after = (
                                record.trust_value
                            )

                            blacklisted = (
                                record.blacklisted
                            )

                            break

                # ------------------------------------------------
                # Determine result.
                # ------------------------------------------------

                if dropped:
                    packet_delivered = False
                    result_status = (
                        "REPLAY_DETECTED"
                    )

                elif routing_result.delivered:

                    packet_delivered = True

                    result_status = (
                        "DELIVERED"
                    )

                else:

                    packet_delivered = False

                    result_status = (
                        "RL_ROUTE_FAILURE"
                    )

                results.append(
                    {
                    "packet_id": packet_id,
                    "source_id": source_id,
                    "route": route,
                    "hop_count": routing_result.hop_count,
                    "rl_route_found": routing_result.delivered,
                    "packet_delivered": packet_delivered,
                    "dropped_by_replay": dropped,
                    "malicious_node": (
                        malicious_node
                        if dropped
                        else ""
                    ),
                    "observer_node": (
                        observer_id
                        if observer_id is not None
                        else ""
                    ),
                    "trust_before": trust_before,
                    "trust_after": trust_after,
                    "blacklisted": blacklisted,
                    "result": result_status,
                }
            )

            packet_id += 1

        # ====================================================
        # METRICS
        # ====================================================

        total_packets = len(results)

        delivered_packets = sum(
            1
            for result in results
            if result["packet_delivered"]
        )

        dropped_packets = sum(
            1
            for result in results
            if result["dropped_by_replay"]
        )

        route_failures = sum(
            1
            for result in results
            if (
                not result["packet_delivered"]
                and not result["dropped_by_replay"]
            )
        )

        pdr = (
            delivered_packets
            / total_packets
            * 100
            if total_packets
            else 0.0
        )

        packet_drop_ratio = (
            dropped_packets
            / total_packets
            * 100
            if total_packets
            else 0.0
        )

        # ====================================================
        # FINAL TRUST
        # ====================================================

        malicious_trust_values = []

        for record in trust_manager.records.values():

            if (
                record.neighbour_id
                == malicious_node
            ):

                malicious_trust_values.append(
                    record.trust_value
                )

        if malicious_trust_values:

            minimum_trust = min(
                malicious_trust_values
            )

            maximum_trust = max(
                malicious_trust_values
            )

        else:

            minimum_trust = 0.0
            maximum_trust = 0.0

        blacklisted_relationships = sum(
            1
            for record in trust_manager.records.values()
            if (
                record.neighbour_id
                == malicious_node
                and record.blacklisted
            )
        )

        # ====================================================
        # OUTPUT
        # ====================================================

        output_path = (
            Path(config.output.directory)
            / "replay_attack_results.csv"
        )

        export_attack_results(
            results=results,
            path=output_path,
        )

        # ====================================================
        # PRINT RESULTS
        # ====================================================

        print("\nReplay attack results")

        print(
            f"  Malicious node          : "
            f"{malicious_node}"
        )

        print(
            f"  Packets generated       : "
            f"{total_packets}"
        )

        print(
            f"  Packets delivered       : "
            f"{delivered_packets}"
        )

        print(
            f"  Packets dropped         : "
            f"{dropped_packets}"
        )

        print(
            f"  Route failures          : "
            f"{route_failures}"
        )

        print(
            f"  Packet delivery ratio   : "
            f"{pdr:.2f}%"
        )

        print(
            f"  Packet drop ratio       : "
            f"{packet_drop_ratio:.2f}%"
        )

        print(
            f"  Minimum malicious trust : "
            f"{minimum_trust:.4f}"
        )

        print(
            f"  Maximum malicious trust : "
            f"{maximum_trust:.4f}"
        )

        print(
            f"  Blacklisted relationships: "
            f"{blacklisted_relationships}"
        )
        print(
            f"  Replay events           : "
            f"{attack.replay_count}"
        )

        # ====================================================
        # ATTACK EVENTS
        # ====================================================

        print(
            "\nTrust/blacklist events"
        )

        if attack.events:

            for event in attack.events[:10]:

                print(
                    f"  Packet {event.packet_id:02d}: "
                    f"{event.observer_id} -> "
                    f"{event.malicious_node_id} | "
                    f"trust "
                    f"{event.trust_before:.4f} -> "
                    f"{event.trust_after:.4f} | "
                    f"blacklisted="
                    f"{event.blacklisted}"
                )

        else:

            print(
                "  No replay event "
                "was observed."
            )

        # ====================================================
        # SAMPLE ROUTES
        # ====================================================

        print(
            "\nSample attack routes"
        )

        for result in results[:10]:

            print(
                f"  Packet "
                f"{result['packet_id']:02d}: "
                f"{' -> '.join(map(str, result['route']))} "
                f"[{result['result']}]"
            )

        # ====================================================
        # VALIDATION
        # ====================================================

        attack_detected = (
            dropped_packets > 0
        )

        blacklist_triggered = (
            blacklisted_relationships > 0
        )

        print(
            "\nPhase 10 validation"
        )

        print(
            f"  Replay attack observed : "
            f"{'PASSED' if attack_detected else 'FAILED'}"
        )

        print(
            f"  Trust degradation         : "
            f"{'PASSED' if attack.events else 'FAILED'}"
        )

        print(
            f"  Blacklisting triggered    : "
            f"{'PASSED' if blacklist_triggered else 'FAILED'}"
        )

        print(
            "\nOutput file:"
        )

        print(
            f"  {output_path}"
        )

        if (
            attack_detected
            and blacklist_triggered
        ):

            print(
                "\nPhase 10 completed successfully."
            )

        else:

            print(
                "\nPhase 10 requires further validation."
            )

        print("=" * 72)

    except (
        ValueError,
        FileNotFoundError,
        KeyError,
        RuntimeError,
    ) as error:

        print("=" * 72)
        print("PHASE 10 ERROR")
        print("=" * 72)

        print(error)

        raise SystemExit(1) from error


if __name__ == "__main__":
    main()