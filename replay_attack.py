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
        seed: int = 149,
    ):

        self.malicious_node_id = malicious_node_id

        self.trust_manager = trust_manager

        self.replay_probability = replay_probability

        # Dedicated random generator so the experiment
        # is reproducible.
        self.random = random.Random(seed)

        self.packet_history: list[int] = []

        self.events: list[ReplayEvent] = []

    def process_hop(
        self,
        packet_id: int,
        observer_id: int,
        next_node_id: int,
    ) -> bool:

        # Only the malicious node performs the replay attack.
        if next_node_id != self.malicious_node_id:
            return True

        # Store the current packet so it can become
        # a replay candidate for later packets.
        self.packet_history.append(packet_id)

        # A replay cannot occur until an older packet exists.
        if len(self.packet_history) < 2:
            return True

        # Apply the configured replay probability.
        if (
            self.random.random()
            > self.replay_probability
        ):
            return True

        # Select an older packet, never the current packet.
        replayed_packet = self.random.choice(
            self.packet_history[:-1]
        )

        trust_before = (
            self.trust_manager.get_trust(
                observer_id,
                next_node_id,
            )
        )

        # Replay is detected and forwarding fails.
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
    def replay_count(self) -> int:

        return len(self.events)