import os

from crypto_module import CryptoModule, SecurePacket
from packet import NetworkPacket


def create_test_packet(sequence_number: int = 1) -> NetworkPacket:
    """
    Create a sample underwater sensor packet for validation.
    """
    return NetworkPacket(
        source=7,
        destination=1,
        sequence_number=sequence_number,
        residual_energy=0.82,
        link_quality=0.91,
        trust_value=0.88,
        distance_to_sink=0.35,
        payload="temperature=24.6",
    )


def print_result(test_name: str, passed: bool, message: str) -> None:
    status = "PASS ✅" if passed else "FAIL ❌"

    print(f"\n{test_name}")
    print(f"Result: {status}")
    print(f"Details: {message}")


def test_normal_packet(
    sender_crypto: CryptoModule,
    receiver_crypto: CryptoModule,
) -> None:
    """
    Test 1:
    A valid packet should be authenticated and decrypted successfully.
    """
    original_packet = create_test_packet(sequence_number=1)

    secure_packet = sender_crypto.encrypt_and_authenticate(
        original_packet.to_bytes()
    )

    try:
        decrypted_data = receiver_crypto.verify_and_decrypt(
            secure_packet
        )

        recovered_packet = NetworkPacket.from_bytes(
            decrypted_data
        )

        passed = recovered_packet == original_packet

        print_result(
            test_name="Test 1: Normal packet transmission",
            passed=passed,
            message=(
                "Packet successfully authenticated and recovered."
                if passed
                else "Recovered packet does not match the original."
            ),
        )

    except ValueError as error:
        print_result(
            test_name="Test 1: Normal packet transmission",
            passed=False,
            message=str(error),
        )


def test_modified_ciphertext(
    sender_crypto: CryptoModule,
    receiver_crypto: CryptoModule,
) -> None:
    """
    Test 2:
    Modify one byte of the ciphertext.

    The existing MAC will no longer match the modified ciphertext,
    so authentication must fail.
    """
    packet = create_test_packet(sequence_number=2)

    secure_packet = sender_crypto.encrypt_and_authenticate(
        packet.to_bytes()
    )

    modified_ciphertext = bytearray(
        secure_packet.ciphertext
    )

    # Change the first byte by flipping its lowest bit.
    modified_ciphertext[0] ^= 1

    tampered_packet = SecurePacket(
        iv=secure_packet.iv,
        ciphertext=bytes(modified_ciphertext),
        mac=secure_packet.mac,
    )

    try:
        receiver_crypto.verify_and_decrypt(tampered_packet)

        print_result(
            test_name="Test 2: Modified ciphertext",
            passed=False,
            message="Tampered ciphertext was incorrectly accepted.",
        )

    except ValueError as error:
        print_result(
            test_name="Test 2: Modified ciphertext",
            passed=True,
            message=str(error),
        )


def test_modified_mac(
    sender_crypto: CryptoModule,
    receiver_crypto: CryptoModule,
) -> None:
    """
    Test 3:
    Modify one byte of the HMAC value.

    The receiver should reject the packet because the supplied MAC
    does not match the locally calculated MAC.
    """
    packet = create_test_packet(sequence_number=3)

    secure_packet = sender_crypto.encrypt_and_authenticate(
        packet.to_bytes()
    )

    # Convert immutable bytes into a mutable bytearray.
    modified_mac = bytearray(secure_packet.mac)

    # Flip one bit in the first MAC byte.
    modified_mac[0] ^= 1

    tampered_packet = SecurePacket(
        iv=secure_packet.iv,
        ciphertext=secure_packet.ciphertext,
        mac=bytes(modified_mac),
    )

    try:
        receiver_crypto.verify_and_decrypt(tampered_packet)

        print_result(
            test_name="Test 3: Modified HMAC",
            passed=False,
            message="Modified HMAC was incorrectly accepted.",
        )

    except ValueError as error:
        print_result(
            test_name="Test 3: Modified HMAC",
            passed=True,
            message=str(error),
        )


def test_wrong_hmac_key(
    aes_key: bytes,
    correct_hmac_key: bytes,
) -> None:
    """
    Test 4:
    Sender and receiver use the same AES key but different HMAC keys.

    Authentication should fail before decryption.
    """
    wrong_hmac_key = os.urandom(32)

    sender_crypto = CryptoModule(
        encryption_key=aes_key,
        hmac_key=correct_hmac_key,
    )

    receiver_crypto = CryptoModule(
        encryption_key=aes_key,
        hmac_key=wrong_hmac_key,
    )

    packet = create_test_packet(sequence_number=4)

    secure_packet = sender_crypto.encrypt_and_authenticate(
        packet.to_bytes()
    )

    try:
        receiver_crypto.verify_and_decrypt(secure_packet)

        print_result(
            test_name="Test 4: Wrong HMAC key",
            passed=False,
            message="Packet was accepted using an incorrect HMAC key.",
        )

    except ValueError as error:
        print_result(
            test_name="Test 4: Wrong HMAC key",
            passed=True,
            message=str(error),
        )


def test_wrong_aes_key(
    correct_aes_key: bytes,
    hmac_key: bytes,
) -> None:
    """
    Test 5:
    Sender and receiver use the same HMAC key but different AES keys.

    HMAC verification succeeds because the HMAC key is correct.
    AES decryption should then produce invalid plaintext or padding.
    """
    wrong_aes_key = os.urandom(16)

    sender_crypto = CryptoModule(
        encryption_key=correct_aes_key,
        hmac_key=hmac_key,
    )

    receiver_crypto = CryptoModule(
        encryption_key=wrong_aes_key,
        hmac_key=hmac_key,
    )

    packet = create_test_packet(sequence_number=5)

    secure_packet = sender_crypto.encrypt_and_authenticate(
        packet.to_bytes()
    )

    try:
        decrypted_data = receiver_crypto.verify_and_decrypt(
            secure_packet
        )

        # In the unlikely event that padding happens to be valid,
        # parsing the random plaintext should still fail.
        NetworkPacket.from_bytes(decrypted_data)

        print_result(
            test_name="Test 5: Wrong AES key",
            passed=False,
            message="Packet was recovered using an incorrect AES key.",
        )

    except (ValueError, UnicodeDecodeError, TypeError) as error:
        print_result(
            test_name="Test 5: Wrong AES key",
            passed=True,
            message=f"Decryption failed as expected: {error}",
        )


def test_modified_iv(
    sender_crypto: CryptoModule,
    receiver_crypto: CryptoModule,
) -> None:
    """
    Test 6:
    Modify the IV.

    The IV is included in HMAC calculation, so changing it must cause
    authentication failure.
    """
    packet = create_test_packet(sequence_number=6)

    secure_packet = sender_crypto.encrypt_and_authenticate(
        packet.to_bytes()
    )

    modified_iv = bytearray(secure_packet.iv)
    modified_iv[0] ^= 1

    tampered_packet = SecurePacket(
        iv=bytes(modified_iv),
        ciphertext=secure_packet.ciphertext,
        mac=secure_packet.mac,
    )

    try:
        receiver_crypto.verify_and_decrypt(tampered_packet)

        print_result(
            test_name="Test 6: Modified IV",
            passed=False,
            message="Packet with a modified IV was accepted.",
        )

    except ValueError as error:
        print_result(
            test_name="Test 6: Modified IV",
            passed=True,
            message=str(error),
        )


def receive_with_replay_detection(
    secure_packet: SecurePacket,
    receiver_crypto: CryptoModule,
    received_sequences: set[int],
) -> NetworkPacket:
    """
    Verify, decrypt, and then check whether the packet's sequence
    number has already been received.
    """
    decrypted_data = receiver_crypto.verify_and_decrypt(
        secure_packet
    )

    packet = NetworkPacket.from_bytes(decrypted_data)

    replay_identifier = (
        packet.source,
        packet.sequence_number,
    )

    if replay_identifier in received_sequences:
        raise ValueError(
            "Replay attack detected: sequence number already received."
        )

    received_sequences.add(replay_identifier)

    return packet


def test_replay_attack(
    sender_crypto: CryptoModule,
    receiver_crypto: CryptoModule,
) -> None:
    """
    Test 7:
    Send exactly the same secure packet twice.

    The first transmission should be accepted.
    The second transmission should be rejected as a replay.
    """
    packet = create_test_packet(sequence_number=100)

    secure_packet = sender_crypto.encrypt_and_authenticate(
        packet.to_bytes()
    )

    received_sequences: set[tuple[int, int]] = set()

    try:
        first_packet = receive_with_replay_detection(
            secure_packet=secure_packet,
            receiver_crypto=receiver_crypto,
            received_sequences=received_sequences,
        )

        print(
            "\nTest 7: Replay attack"
        )
        print(
            "First transmission: ACCEPTED ✅ "
            f"(sequence number {first_packet.sequence_number})"
        )

    except ValueError as error:
        print_result(
            test_name="Test 7: Replay attack",
            passed=False,
            message=f"First transmission failed: {error}",
        )
        return

    try:
        receive_with_replay_detection(
            secure_packet=secure_packet,
            receiver_crypto=receiver_crypto,
            received_sequences=received_sequences,
        )

        print("Second transmission: ACCEPTED ❌")
        print("Result: FAIL ❌")
        print("Details: Replayed packet was not detected.")

    except ValueError as error:
        print("Second transmission: REJECTED ✅")
        print("Result: PASS ✅")
        print(f"Details: {error}")


def main() -> None:
    # AES-128 requires a 16-byte key.
    aes_key = os.urandom(16)

    # HMAC-SHA256 key.
    hmac_key = os.urandom(32)

    sender_crypto = CryptoModule(
        encryption_key=aes_key,
        hmac_key=hmac_key,
    )

    receiver_crypto = CryptoModule(
        encryption_key=aes_key,
        hmac_key=hmac_key,
    )

    print("=" * 55)
    print("AES-128 + HMAC-SHA256 SECURITY VALIDATION")
    print("=" * 55)

    test_normal_packet(
        sender_crypto,
        receiver_crypto,
    )

    test_modified_ciphertext(
        sender_crypto,
        receiver_crypto,
    )

    test_modified_mac(
        sender_crypto,
        receiver_crypto,
    )

    test_wrong_hmac_key(
        aes_key=aes_key,
        correct_hmac_key=hmac_key,
    )

    test_wrong_aes_key(
        correct_aes_key=aes_key,
        hmac_key=hmac_key,
    )

    test_modified_iv(
        sender_crypto,
        receiver_crypto,
    )

    test_replay_attack(
        sender_crypto,
        receiver_crypto,
    )


if __name__ == "__main__":
    main()