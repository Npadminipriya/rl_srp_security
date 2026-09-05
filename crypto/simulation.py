import os
import time

from crypto_module import CryptoModule, SecurePacket
from packet import NetworkPacket
from metrics import SimulationMetrics


def calculate_secured_size(packet: SecurePacket) -> int:
    return (
        len(packet.iv)
        + len(packet.ciphertext)
        + len(packet.mac)
    )


def run_normal_packet_test(
    crypto: CryptoModule,
    metrics: SimulationMetrics,
    sequence_number: int,
) -> None:
    packet = NetworkPacket(
        source=7,
        destination=1,
        sequence_number=sequence_number,
        residual_energy=0.82,
        link_quality=0.91,
        trust_value=0.88,
        distance_to_sink=0.35,
        payload="temperature=24.6",
    )

    plaintext = packet.to_bytes()

    start = time.perf_counter()
    secure_packet = crypto.encrypt_and_authenticate(plaintext)
    encryption_time = time.perf_counter() - start

    start = time.perf_counter()
    decrypted_data = crypto.verify_and_decrypt(secure_packet)
    decryption_time = time.perf_counter() - start

    recovered_packet = NetworkPacket.from_bytes(decrypted_data)

    print("Original packet:", packet)
    print("Recovered packet:", recovered_packet)

    metrics.record_success(
        encryption_time=encryption_time,
        decryption_time=decryption_time,
        original_size=len(plaintext),
        secured_size=calculate_secured_size(secure_packet),
    )


def run_tampering_test(
    crypto: CryptoModule,
    metrics: SimulationMetrics,
) -> None:
    packet = NetworkPacket(
        source=4,
        destination=1,
        sequence_number=200,
        residual_energy=0.65,
        link_quality=0.79,
        trust_value=0.72,
        distance_to_sink=0.44,
        payload="pressure=18.3",
    )

    secure_packet = crypto.encrypt_and_authenticate(
        packet.to_bytes()
    )

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
        crypto.verify_and_decrypt(tampered_packet)
        print("Warning: tampered packet was accepted.")
    except ValueError as error:
        print("Tampered packet rejected:", error)
        metrics.record_rejection()


def main() -> None:
    aes_key = os.urandom(16)
    hmac_key = os.urandom(32)

    crypto = CryptoModule(
        encryption_key=aes_key,
        hmac_key=hmac_key,
    )

    metrics = SimulationMetrics()

    for sequence_number in range(1, 6):
        run_normal_packet_test(
            crypto=crypto,
            metrics=metrics,
            sequence_number=sequence_number,
        )

    run_tampering_test(
        crypto=crypto,
        metrics=metrics,
    )

    metrics.print_summary()


if __name__ == "__main__":
    main()