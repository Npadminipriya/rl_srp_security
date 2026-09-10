from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AttackEvent:
    packet_id: int
    malicious_node_id: int
    observer_id: int
    attack_type: str
    dropped: bool
    trust_before: float
    trust_after: float
    blacklisted: bool


class BlackholeAttack:
    """
    Blackhole attack model.

    A malicious node behaves normally from the routing
    perspective but drops every packet that reaches it.

    The previous node observes the forwarding failure,
    causing the malicious node's trust value to decrease.
    """

    def __init__(
        self,
        malicious_node_id: int,
        trust_manager,
    ) -> None:

        self.malicious_node_id = malicious_node_id
        self.trust_manager = trust_manager

        self.events: list[AttackEvent] = []

    # ========================================================
    # CHECK WHETHER NODE IS MALICIOUS
    # ========================================================

    def is_malicious(
        self,
        node_id: int,
    ) -> bool:

        return (
            node_id == self.malicious_node_id
        )

    # ========================================================
    # PROCESS FORWARDING
    # ========================================================

    def process_hop(
        self,
        packet_id: int,
        observer_id: int,
        next_node_id: int,
    ) -> bool:
        """
        Simulate one forwarding attempt.

        Returns:

            True  -> packet successfully forwarded
            False -> packet dropped by blackhole
        """

        # ----------------------------------------------------
        # Normal node
        # ----------------------------------------------------

        if not self.is_malicious(next_node_id):

            return True

        # ----------------------------------------------------
        # Blackhole node
        # ----------------------------------------------------

        trust_before = (
            self.trust_manager.get_trust(
                observer_id,
                next_node_id,
            )
        )

        # Blackhole drops packet.
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
                attack_type="BLACKHOLE",
                dropped=True,
                trust_before=trust_before,
                trust_after=record.trust_value,
                blacklisted=record.blacklisted,
            )
        )

        return False

    # ========================================================
    # SUMMARY
    # ========================================================

    @property
    def packets_dropped(self) -> int:

        return sum(
            1
            for event in self.events
            if event.dropped
        )

    @property
    def blacklist_events(self) -> int:

        return sum(
            1
            for event in self.events
            if event.blacklisted
        )

    def clear_events(self) -> None:

        self.events.clear()