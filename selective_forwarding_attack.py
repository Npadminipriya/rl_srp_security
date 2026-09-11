from __future__ import annotations

import random

from dataclasses import dataclass


@dataclass
class AttackEvent:
    """
    Records a selective forwarding attack event.
    """

    packet_id: int
    malicious_node_id: int
    observer_id: int

    attack_type: str

    dropped: bool

    trust_before: float
    trust_after: float

    blacklisted: bool


class SelectiveForwardingAttack:
    """
    Selective Forwarding Attack.

    A malicious node drops only a percentage
    of packets and forwards the rest.
    """

    def __init__(
        self,
        malicious_node_id: int,
        trust_manager,
        drop_probability: float = 0.5,
    ) -> None:

        self.malicious_node_id = malicious_node_id
        self.trust_manager = trust_manager

        self.drop_probability = drop_probability

        self.events: list[AttackEvent] = []

    def is_malicious(
        self,
        node_id: int,
    ) -> bool:

        return (
            node_id == self.malicious_node_id
        )

    def process_hop(
        self,
        packet_id: int,
        observer_id: int,
        next_node_id: int,
    ) -> bool:
        """
        Returns:

            True  -> packet forwarded

            False -> packet dropped
        """

        if not self.is_malicious(
            next_node_id
        ):
            return True

        trust_before = (
            self.trust_manager.get_trust(
                observer_id,
                next_node_id,
            )
        )

        drop_packet = (
            random.random()
            < self.drop_probability
        )

        if drop_packet:

            record = (
                self.trust_manager.record_forwarding(
                    observer_id=observer_id,
                    neighbour_id=next_node_id,
                    success=False,
                )
            )

            self.events.append(
                AttackEvent(
                    packet_id=packet_id,
                    malicious_node_id=next_node_id,
                    observer_id=observer_id,
                    attack_type="SELECTIVE_FORWARDING",
                    dropped=True,
                    trust_before=trust_before,
                    trust_after=record.trust_value,
                    blacklisted=record.blacklisted,
                )
            )

            return False

        record = (
            self.trust_manager.record_forwarding(
                observer_id=observer_id,
                neighbour_id=next_node_id,
                success=True,
            )
        )

        self.events.append(
            AttackEvent(
                packet_id=packet_id,
                malicious_node_id=next_node_id,
                observer_id=observer_id,
                attack_type="SELECTIVE_FORWARDING",
                dropped=False,
                trust_before=trust_before,
                trust_after=record.trust_value,
                blacklisted=record.blacklisted,
            )
        )

        return True

    @property
    def packets_dropped(
        self,
    ) -> int:

        return sum(
            1
            for event in self.events
            if event.dropped
        )

    @property
    def packets_forwarded(
        self,
    ) -> int:

        return sum(
            1
            for event in self.events
            if not event.dropped
        )

    @property
    def blacklist_events(
        self,
    ) -> int:

        return sum(
            1
            for event in self.events
            if event.blacklisted
        )

    def clear_events(
        self,
    ) -> None:

        self.events.clear()