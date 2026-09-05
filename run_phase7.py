from __future__ import annotations

import csv
import os
from pathlib import Path

from config import load_config
from crypto.crypto_module import CryptoModule

from q_learning import QLearningRouter
from secure_routing import SecureRoutingEngine
from topology import load_topology_snapshot
from trust_manager import TrustManager


def export_results(
    results,
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
                "rl_delivered",
                "secure_delivered",
                "encrypted",
                "hmac_verified",
                "decrypted",
                "payload_recovered",
                "original_size_bytes",
                "secured_size_bytes",
                "overhead_bytes",
                "encryption_time_ms",
                "verification_decryption_time_ms",
                "security_status",
                "reason",
            ]
        )

        for result in results:

            writer.writerow(
                [
                    result.packet_id,
                    result.source_id,
                    "->".join(
                        map(str, result.route)
                    ),
                    result.hop_count,
                    result.rl_delivered,
                    result.secure_delivered,
                    result.encrypted,
                    result.hmac_verified,
                    result.decrypted,
                    result.payload_recovered,
                    result.original_size,
                    result.secured_size,
                    (
                        result.secured_size
                        - result.original_size
                    ),
                    f"{result.encryption_time_ms:.6f}",
                    (
                        f"{result.verification_decryption_time_ms:.6f}"
                    ),
                    result.security_status,
                    result.reason,
                ]
            )


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

        # ====================================================
        # CREATE CRYPTOGRAPHIC KEYS
        # ====================================================

        # AES-128 = 16 bytes
        aes_key = os.urandom(16)

        # HMAC-SHA256 = 32-byte key
        hmac_key = os.urandom(32)

        sender_crypto = CryptoModule(
            encryption_key=aes_key,
            hmac_key=hmac_key,
        )

        receiver_crypto = CryptoModule(
            encryption_key=aes_key,
            hmac_key=hmac_key,
        )

        # ====================================================
        # CREATE RL ROUTER
        # ====================================================

        router = QLearningRouter(
            topology=topology,
            config=config,
            trust_manager=trust_manager,
        )

        print("=" * 72)
        print(
            "RL-SRP PHASE 7: SECURE PACKET ROUTING"
        )
        print("=" * 72)

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
        # CREATE SECURITY ENGINE
        # ====================================================

        engine = SecureRoutingEngine(
            router=router,
            trust_manager=trust_manager,
            sender_crypto=sender_crypto,
            receiver_crypto=receiver_crypto,
        )

        # ====================================================
        # SECURE ROUTING TEST
        # ====================================================

        print(
            "\nTesting secure packet transmission..."
        )

        results = []

        packet_id = 1

        for node in topology.nodes:

            if node.node_type != "sensor":
                continue

            payload = (
                f"sensor={node.node_id};"
                f"temperature={20.0 + node.node_id * 0.1:.2f};"
                f"pressure={1000.0 + node.node_id:.2f}"
            )

            result = engine.transmit_packet(
                packet_id=packet_id,
                source_id=node.node_id,
                payload=payload,
            )

            results.append(result)

            packet_id += 1

        # ====================================================
        # METRICS
        # ====================================================

        total = len(results)

        rl_delivered = sum(
            result.rl_delivered
            for result in results
        )

        secure_delivered = sum(
            result.secure_delivered
            for result in results
        )

        encrypted = sum(
            result.encrypted
            for result in results
        )

        hmac_verified = sum(
            result.hmac_verified
            for result in results
        )

        decrypted = sum(
            result.decrypted
            for result in results
        )

        recovered = sum(
            result.payload_recovered
            for result in results
        )

        average_hops = (
            sum(
                result.hop_count
                for result in results
                if result.rl_delivered
            )
            / rl_delivered
            if rl_delivered
            else 0.0
        )

        average_encryption = (
            sum(
                result.encryption_time_ms
                for result in results
            )
            / total
            if total
            else 0.0
        )

        average_verification = (
            sum(
                result.verification_decryption_time_ms
                for result in results
            )
            / total
            if total
            else 0.0
        )

        average_overhead = (
            sum(
                result.secured_size
                - result.original_size
                for result in results
            )
            / total
            if total
            else 0.0
        )

        # ====================================================
        # OUTPUT
        # ====================================================

        print("\nSecure routing results")

        print(
            f"  Packets generated       : {total}"
        )

        print(
            f"  RL packets delivered    : "
            f"{rl_delivered}"
        )

        print(
            f"  Secure packets delivered: "
            f"{secure_delivered}"
        )

        print(
            f"  RL delivery ratio       : "
            f"{rl_delivered / total * 100:.2f}%"
        )

        print(
            f"  Secure delivery ratio   : "
            f"{secure_delivered / total * 100:.2f}%"
        )

        print(
            f"  AES encryption          : "
            f"{encrypted}/{total}"
        )

        print(
            f"  HMAC verification       : "
            f"{hmac_verified}/{total}"
        )

        print(
            f"  AES decryption          : "
            f"{decrypted}/{total}"
        )

        print(
            f"  Payload recovery        : "
            f"{recovered}/{total}"
        )

        print(
            f"  Average hop count       : "
            f"{average_hops:.2f}"
        )

        print(
            f"  Avg encryption time     : "
            f"{average_encryption:.6f} ms"
        )

        print(
            f"  Avg verify/decrypt time : "
            f"{average_verification:.6f} ms"
        )

        print(
            f"  Avg security overhead   : "
            f"{average_overhead:.2f} bytes"
        )

        # ====================================================
        # TAMPERING TEST
        # ====================================================

        print(
            "\nTampering integration test"
        )

        tampering_passed = (
            engine.tampering_test(
                packet_id=9999,
                source_id=1,
                payload="tamper-test",
            )
        )

        print(
            "  Modified ciphertext rejected: "
            f"{'PASSED' if tampering_passed else 'FAILED'}"
        )

        # ====================================================
        # SAMPLE ROUTES
        # ====================================================

        print("\nSample secure routes")

        for result in results[:10]:

            route = " -> ".join(
                map(str, result.route)
            )

            status = (
                "SECURE DELIVERY"
                if result.secure_delivered
                else result.security_status
            )

            print(
                f"  Packet {result.packet_id:02d}: "
                f"{route} [{status}]"
            )

        # ====================================================
        # EXPORT
        # ====================================================

        output_path = (
            Path(config.output.directory)
            / "secure_rl_srp_results.csv"
        )

        export_results(
            results=results,
            path=output_path,
        )

        print(
            "\nOutput file:"
        )

        print(
            f"  {output_path}"
        )

        # ====================================================
        # FINAL VALIDATION
        # ====================================================

        all_secure = (
            secure_delivered == total
        )

        if (
            all_secure
            and tampering_passed
        ):

            print(
                "\nPhase 7 completed successfully."
            )

        else:

            print(
                "\nPhase 7 completed with validation issues."
            )

        print("=" * 72)

    except (
        ValueError,
        FileNotFoundError,
        KeyError,
    ) as error:

        print("=" * 72)
        print("PHASE 7 ERROR")
        print("=" * 72)

        print(error)

        raise SystemExit(1) from error


if __name__ == "__main__":
    main()