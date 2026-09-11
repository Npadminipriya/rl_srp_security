from __future__ import annotations

import random

from dataclasses import dataclass


@dataclass
class ModificationEvent:

    packet_id: int

    observer_id: int

    malicious_node_id: int

    trust_before: float

    trust_after: float

    blacklisted: bool


class ModificationAttack:

    def __init__(
        self,
        malicious_node_id: int,
        trust_manager,
        modification_probability: float = 0.30,
    ):

        self.malicious_node_id = malicious_node_id

        self.trust_manager = trust_manager

        self.modification_probability = modification_probability

        self.events = []

    def process_hop(
        self,
        packet_id: int,
        observer_id: int,
        next_node_id: int,
    ) -> bool:

        if next_node_id != self.malicious_node_id:
            return True

        if random.random() > self.modification_probability:
            return True

        trust_before = (
            self.trust_manager.get_trust(
                observer_id,
                next_node_id,
            )
        )

        record = (
            self.trust_manager.record_forwarding(
                observer_id=observer_id,
                neighbour_id=next_node_id,
                success=False,
            )
        )

        self.events.append(
            ModificationEvent(
                packet_id=packet_id,
                observer_id=observer_id,
                malicious_node_id=next_node_id,
                trust_before=trust_before,
                trust_after=record.trust_value,
                blacklisted=record.blacklisted,
            )
        )

        return False

    @property
    def modification_count(self):

        return len(self.events)