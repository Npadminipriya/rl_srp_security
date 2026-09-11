from __future__ import annotations

import random

from dataclasses import dataclass


@dataclass
class ReplayEvent:

    packet_id: int

    replayed_packet_id: int

    observer_id: int

    malicious_node_id: int

    trust_before: float

    trust_after: float

    blacklisted: bool


class ReplayAttack:

    def __init__(
        self,
        malicious_node_id: int,
        trust_manager,
        replay_probability: float = 0.30,
    ):

        self.malicious_node_id = malicious_node_id

        self.trust_manager = trust_manager

        self.replay_probability = replay_probability

        self.packet_history = []

        self.events = []

    def process_hop(
        self,
        packet_id: int,
        observer_id: int,
        next_node_id: int,
    ) -> bool:

        if next_node_id != self.malicious_node_id:
            return True

        self.packet_history.append(packet_id)

        if len(self.packet_history) < 2:
            return True

        if random.random() > self.replay_probability:
            return True

        replayed_packet = random.choice(
            self.packet_history[:-1]
        )

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
            ReplayEvent(
                packet_id=packet_id,
                replayed_packet_id=replayed_packet,
                observer_id=observer_id,
                malicious_node_id=next_node_id,
                trust_before=trust_before,
                trust_after=record.trust_value,
                blacklisted=record.blacklisted,
            )
        )

        return False

    @property
    def replay_count(self):

        return len(self.events)