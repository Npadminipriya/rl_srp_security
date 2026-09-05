from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from crypto.crypto_module import CryptoModule, SecurePacket
from crypto.packet import NetworkPacket

from node import Node
from q_learning import QLearningRouter, RLRoutingResult
from trust_manager import TrustManager


@dataclass
class SecureRoutingResult:
    packet_id: int
    source_id: int

    route: list[int]
    hop_count: int

    rl_delivered: bool
    secure_delivered: bool

    encrypted: bool
    hmac_verified: bool
    decrypted: bool
    payload_recovered: bool

    original_size: int
    secured_size: int

    encryption_time_ms: float
    verification_decryption_time_ms: float

    security_status: str
    reason: str


class SecureRoutingEngine:
    """
    Phase 7 integration layer.

    Connects:

        RL-SRP routing
              +
        AES-128 encryption
              +
        HMAC-SHA256 authentication
    """

    def __init__(
        self,
        router: QLearningRouter,
        trust_manager: TrustManager,
        sender_crypto: CryptoModule,
        receiver_crypto: CryptoModule,
    ) -> None:

        self.router = router
        self.trust_manager = trust_manager

        self.sender_crypto = sender_crypto
        self.receiver_crypto = receiver_crypto

        self.received_sequences: set[
            tuple[int, int]
        ] = set()

    # ========================================================
    # BUILD NETWORK PACKET
    # ========================================================

    def build_network_packet(
        self,
        packet_id: int,
        source_id: int,
        payload: str,
    ) -> NetworkPacket:

        source = self.router.node_by_id[source_id]
        sink = self.router.sink

        # ----------------------------------------------------
        # Obtain initial routing information.
        # ----------------------------------------------------

        trust_value = self.router.config.trust.initial_value

        # Use the first available neighbour's trust
        # when one exists.
        if source.neighbours:

            for neighbour_id in source.neighbours:

                if not self.trust_manager.is_blacklisted(
                    source_id,
                    neighbour_id,
                ):

                    trust_value = (
                        self.trust_manager.get_trust(
                            source_id,
                            neighbour_id,
                        )
                    )

                    break

        # ----------------------------------------------------
        # Link quality approximation.
        #
        # The routing environment already calculates link
        # quality from distance. For the secure packet we
        # store the source-level value as metadata.
        # ----------------------------------------------------

        link_quality = 1.0

        if source.neighbours:

            distances = []

            for neighbour_id in source.neighbours:

                neighbour = (
                    self.router.node_by_id[
                        neighbour_id
                    ]
                )

                distances.append(
                    source.distance_to(neighbour)
                )

            if distances:

                minimum_distance = min(
                    distances
                )

                communication_range = (
                    self.router.config.topology
                    .communication_range_m
                )

                link_quality = max(
                    0.0,
                    min(
                        1.0,
                        1.0
                        - (
                            minimum_distance
                            / communication_range
                        ),
                    ),
                )

        distance_to_sink = source.distance_to_sink(
            sink
        )

        # Normalize distance for storage.
        max_distance = (
            (
                self.router.config.topology.area_x_m
                ** 2
                +
                self.router.config.topology.area_y_m
                ** 2
                +
                self.router.config.topology.depth_m
                ** 2
            )
            ** 0.5
        )

        normalized_distance = (
            distance_to_sink / max_distance
        )

        residual_energy = (
            source.residual_energy_j
            if source.residual_energy_j is not None
            else 0.0
        )

        initial_energy = (
            source.initial_energy_j
            if source.initial_energy_j is not None
            else 1.0
        )

        residual_energy_ratio = (
            residual_energy / initial_energy
            if initial_energy > 0
            else 0.0
        )

        return NetworkPacket(
            source=source_id,
            destination=0,
            sequence_number=packet_id,
            residual_energy=residual_energy_ratio,
            link_quality=link_quality,
            trust_value=trust_value,
            distance_to_sink=normalized_distance,
            payload=payload,
        )

    # ========================================================
    # SECURE ONE PACKET
    # ========================================================

    def transmit_packet(
        self,
        packet_id: int,
        source_id: int,
        payload: str,
    ) -> SecureRoutingResult:

        # ----------------------------------------------------
        # STEP 1 — RL routing
        # ----------------------------------------------------

        routing_result = self.router.route_packet(
            packet_id=packet_id,
            source_id=source_id,
        )

        if not routing_result.delivered:

            return SecureRoutingResult(
                packet_id=packet_id,
                source_id=source_id,
                route=routing_result.route,
                hop_count=routing_result.hop_count,
                rl_delivered=False,
                secure_delivered=False,
                encrypted=False,
                hmac_verified=False,
                decrypted=False,
                payload_recovered=False,
                original_size=0,
                secured_size=0,
                encryption_time_ms=0.0,
                verification_decryption_time_ms=0.0,
                security_status="NOT_TRANSMITTED",
                reason=routing_result.reason,
            )

        # ----------------------------------------------------
        # STEP 2 — Construct actual network packet
        # ----------------------------------------------------

        network_packet = self.build_network_packet(
            packet_id=packet_id,
            source_id=source_id,
            payload=payload,
        )

        plaintext = network_packet.to_bytes()

        # ----------------------------------------------------
        # STEP 3 — AES-128 + HMAC-SHA256
        # ----------------------------------------------------

        encryption_start = time.perf_counter()

        secure_packet = (
            self.sender_crypto
            .encrypt_and_authenticate(
                plaintext
            )
        )

        encryption_time = (
            time.perf_counter()
            - encryption_start
        )

        secured_size = (
            len(secure_packet.iv)
            + len(secure_packet.ciphertext)
            + len(secure_packet.mac)
        )

        # ----------------------------------------------------
        # STEP 4 — Simulated forwarding
        #
        # The RL route has already been established.
        # We now treat the secure packet as travelling
        # across that route.
        # ----------------------------------------------------

        forwarded_route = routing_result.route

        # ----------------------------------------------------
        # STEP 5 — Sink authentication + decryption
        # ----------------------------------------------------

        verification_start = time.perf_counter()

        try:

            decrypted_data = (
                self.receiver_crypto
                .verify_and_decrypt(
                    secure_packet
                )
            )

            verification_decryption_time = (
                time.perf_counter()
                - verification_start
            )

            recovered_packet = (
                NetworkPacket.from_bytes(
                    decrypted_data
                )
            )

            # ------------------------------------------------
            # STEP 6 — Replay protection
            # ------------------------------------------------

            replay_identifier = (
                recovered_packet.source,
                recovered_packet.sequence_number,
            )

            if replay_identifier in (
                self.received_sequences
            ):

                return SecureRoutingResult(
                    packet_id=packet_id,
                    source_id=source_id,
                    route=forwarded_route,
                    hop_count=routing_result.hop_count,
                    rl_delivered=True,
                    secure_delivered=False,
                    encrypted=True,
                    hmac_verified=True,
                    decrypted=True,
                    payload_recovered=False,
                    original_size=len(plaintext),
                    secured_size=secured_size,
                    encryption_time_ms=(
                        encryption_time * 1000
                    ),
                    verification_decryption_time_ms=(
                        verification_decryption_time
                        * 1000
                    ),
                    security_status="REPLAY_REJECTED",
                    reason="Replay detected at sink.",
                )

            self.received_sequences.add(
                replay_identifier
            )

            # ------------------------------------------------
            # STEP 7 — Compare recovered packet
            # ------------------------------------------------

            payload_recovered = (
                recovered_packet == network_packet
            )

            return SecureRoutingResult(
                packet_id=packet_id,
                source_id=source_id,
                route=forwarded_route,
                hop_count=routing_result.hop_count,
                rl_delivered=True,
                secure_delivered=payload_recovered,
                encrypted=True,
                hmac_verified=True,
                decrypted=True,
                payload_recovered=payload_recovered,
                original_size=len(plaintext),
                secured_size=secured_size,
                encryption_time_ms=(
                    encryption_time * 1000
                ),
                verification_decryption_time_ms=(
                    verification_decryption_time
                    * 1000
                ),
                security_status=(
                    "SECURE_DELIVERY"
                    if payload_recovered
                    else "PAYLOAD_MISMATCH"
                ),
                reason=(
                    "Packet authenticated and recovered."
                    if payload_recovered
                    else "Recovered packet differs from original."
                ),
            )

        except (
            ValueError,
            UnicodeDecodeError,
            TypeError,
        ) as error:

            verification_decryption_time = (
                time.perf_counter()
                - verification_start
            )

            return SecureRoutingResult(
                packet_id=packet_id,
                source_id=source_id,
                route=forwarded_route,
                hop_count=routing_result.hop_count,
                rl_delivered=True,
                secure_delivered=False,
                encrypted=True,
                hmac_verified=False,
                decrypted=False,
                payload_recovered=False,
                original_size=len(plaintext),
                secured_size=secured_size,
                encryption_time_ms=(
                    encryption_time * 1000
                ),
                verification_decryption_time_ms=(
                    verification_decryption_time
                    * 1000
                ),
                security_status="SECURITY_REJECTED",
                reason=str(error),
            )

    # ========================================================
    # TAMPERING TEST
    # ========================================================

    def tampering_test(
        self,
        packet_id: int,
        source_id: int,
        payload: str,
    ) -> bool:

        network_packet = self.build_network_packet(
            packet_id=packet_id,
            source_id=source_id,
            payload=payload,
        )

        secure_packet = (
            self.sender_crypto
            .encrypt_and_authenticate(
                network_packet.to_bytes()
            )
        )

        # Modify ciphertext after authentication.
        modified_ciphertext = bytearray(
            secure_packet.ciphertext
        )

        modified_ciphertext[0] ^= 1

        tampered_packet = SecurePacket(
            iv=secure_packet.iv,
            ciphertext=bytes(modified_ciphertext),
            mac=secure_packet.mac,
        )

        try:

            self.receiver_crypto.verify_and_decrypt(
                tampered_packet
            )

            return False

        except ValueError:

            return True