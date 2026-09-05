from dataclasses import dataclass, asdict
import json


@dataclass
class NetworkPacket:
    source: int
    destination: int
    sequence_number: int
    residual_energy: float
    link_quality: float
    trust_value: float
    distance_to_sink: float
    payload: str

    def to_bytes(self) -> bytes:
        """
        Convert packet data into bytes before encryption.
        """
        return json.dumps(asdict(self)).encode("utf-8")

    @staticmethod
    def from_bytes(data: bytes) -> "NetworkPacket":
        """
        Convert decrypted bytes back into a NetworkPacket object.
        """
        packet_dict = json.loads(data.decode("utf-8"))
        return NetworkPacket(**packet_dict)