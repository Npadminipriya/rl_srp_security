from __future__ import annotations

import csv

from dataclasses import dataclass
from pathlib import Path

from config import ProjectConfig
from topology import TopologyResult


@dataclass
class TrustRecord:
    """
    Trust maintained by one node about one neighbour.
    """

    observer_id: int
    neighbour_id: int

    trust_value: float

    successful_forwards: int = 0
    total_forwards: int = 0

    blacklisted: bool = False


class TrustManager:
    """
    Decentralized neighbour-trust manager.

    Trust is directional:

        trust[Node A][Node B]

    represents A's opinion of B.
    """

    def __init__(
        self,
        topology: TopologyResult,
        config: ProjectConfig,
    ) -> None:

        self.topology = topology
        self.config = config

        self.records: dict[
            tuple[int, int],
            TrustRecord,
        ] = {}

        self._initialize_records()

    # ------------------------------------------------
    # Initialization
    # ------------------------------------------------

    def _initialize_records(self) -> None:
        """
        Every node initially assigns trust 0.5
        to each of its neighbours.
        """

        initial_trust = (
            self.config.trust.initial_value
        )

        for node in self.topology.nodes:

            for neighbour_id in node.neighbours:

                key = (
                    node.node_id,
                    neighbour_id,
                )

                self.records[key] = TrustRecord(
                    observer_id=node.node_id,
                    neighbour_id=neighbour_id,
                    trust_value=initial_trust,
                )

    # ------------------------------------------------
    # Get trust
    # ------------------------------------------------

    def get_trust(
        self,
        observer_id: int,
        neighbour_id: int,
    ) -> float:

        key = (
            observer_id,
            neighbour_id,
        )

        record = self.records.get(key)

        if record is None:
            raise KeyError(
                f"Node {neighbour_id} is not a neighbour "
                f"of node {observer_id}."
            )

        return record.trust_value

    # ------------------------------------------------
    # Blacklist check
    # ------------------------------------------------

    def is_blacklisted(
        self,
        observer_id: int,
        neighbour_id: int,
    ) -> bool:

        key = (
            observer_id,
            neighbour_id,
        )

        record = self.records.get(key)

        if record is None:
            return True

        return record.blacklisted

    # ------------------------------------------------
    # Trust update
    # ------------------------------------------------

    def record_forwarding(
        self,
        observer_id: int,
        neighbour_id: int,
        success: bool,
    ) -> TrustRecord:
        """
        Update trust after observing forwarding behaviour.

        observed_trust =
            successful forwards / total forwards

        updated_trust =
            lambda * old_trust
            + (1 - lambda) * observed_trust
        """

        key = (
            observer_id,
            neighbour_id,
        )

        if key not in self.records:
            raise KeyError(
                f"No trust relationship exists for "
                f"{observer_id} -> {neighbour_id}."
            )

        record = self.records[key]

        record.total_forwards += 1

        if success:
            record.successful_forwards += 1

        observed_trust = (
            record.successful_forwards
            / record.total_forwards
        )

        memory_factor = (
            self.config.trust.memory_factor
        )

        old_trust = record.trust_value

        new_trust = (
            memory_factor * old_trust
            + (1.0 - memory_factor)
            * observed_trust
        )

        new_trust = max(
            0.0,
            min(1.0, new_trust),
        )

        record.trust_value = new_trust

        threshold = (
            self.config.trust.malicious_threshold
        )

        if record.trust_value < threshold:
            record.blacklisted = True

        return record

    # ------------------------------------------------
    # Reset
    # ------------------------------------------------

    def reset(self) -> None:

        self.records.clear()
        self._initialize_records()

    # ------------------------------------------------
    # Export
    # ------------------------------------------------

    def export_csv(
        self,
        path: str | Path,
    ) -> Path:

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
                    "observer_id",
                    "neighbour_id",
                    "successful_forwards",
                    "total_forwards",
                    "trust_value",
                    "blacklisted",
                ]
            )

            for key in sorted(
                self.records.keys()
            ):

                record = self.records[key]

                writer.writerow(
                    [
                        record.observer_id,
                        record.neighbour_id,
                        record.successful_forwards,
                        record.total_forwards,
                        f"{record.trust_value:.6f}",
                        record.blacklisted,
                    ]
                )

        return output_path