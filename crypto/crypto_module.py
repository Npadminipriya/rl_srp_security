import hashlib
import hmac
import os
from dataclasses import dataclass

from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


@dataclass
class SecurePacket:
    iv: bytes
    ciphertext: bytes
    mac: bytes


class CryptoModule:
    def __init__(self, encryption_key: bytes, hmac_key: bytes) -> None:
        if len(encryption_key) not in (16, 24, 32):
            raise ValueError(
                "AES key must be 16, 24, or 32 bytes "
                "(128, 192, or 256 bits)."
            )

        if len(hmac_key) < 32:
            raise ValueError("Use an HMAC key of at least 32 bytes.")

        self.encryption_key = encryption_key
        self.hmac_key = hmac_key

    def encrypt_and_authenticate(self, plaintext: bytes) -> SecurePacket:
        # AES-CBC requires a fresh 16-byte IV for every encryption.
        iv = os.urandom(16)

        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        padded_plaintext = padder.update(plaintext) + padder.finalize()

        cipher = Cipher(
            algorithms.AES(self.encryption_key),
            modes.CBC(iv),
        )

        encryptor = cipher.encryptor()
        ciphertext = (
            encryptor.update(padded_plaintext)
            + encryptor.finalize()
        )

        # Authenticate both IV and ciphertext.
        authenticated_data = iv + ciphertext

        mac = hmac.new(
            self.hmac_key,
            authenticated_data,
            hashlib.sha256,
        ).digest()

        return SecurePacket(
            iv=iv,
            ciphertext=ciphertext,
            mac=mac,
        )

    def verify_and_decrypt(self, packet: SecurePacket) -> bytes:
        authenticated_data = packet.iv + packet.ciphertext

        expected_mac = hmac.new(
            self.hmac_key,
            authenticated_data,
            hashlib.sha256,
        ).digest()

        # Constant-time comparison.
        if not hmac.compare_digest(packet.mac, expected_mac):
            raise ValueError("Authentication failed: packet was modified.")

        cipher = Cipher(
            algorithms.AES(self.encryption_key),
            modes.CBC(packet.iv),
        )

        decryptor = cipher.decryptor()
        padded_plaintext = (
            decryptor.update(packet.ciphertext)
            + decryptor.finalize()
        )

        unpadder = padding.PKCS7(
            algorithms.AES.block_size
        ).unpadder()

        return (
            unpadder.update(padded_plaintext)
            + unpadder.finalize()
        )