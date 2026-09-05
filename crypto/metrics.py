from dataclasses import dataclass


@dataclass
class SimulationMetrics:
    total_packets: int = 0
    successful_packets: int = 0
    rejected_packets: int = 0
    encryption_time: float = 0.0
    decryption_time: float = 0.0
    original_size: int = 0
    secured_size: int = 0

    def record_success(
        self,
        encryption_time: float,
        decryption_time: float,
        original_size: int,
        secured_size: int,
    ) -> None:
        self.total_packets += 1
        self.successful_packets += 1
        self.encryption_time += encryption_time
        self.decryption_time += decryption_time
        self.original_size += original_size
        self.secured_size += secured_size

    def record_rejection(self) -> None:
        self.total_packets += 1
        self.rejected_packets += 1

    def print_summary(self) -> None:
        if self.successful_packets > 0:
            avg_encryption = (
                self.encryption_time / self.successful_packets
            )
            avg_decryption = (
                self.decryption_time / self.successful_packets
            )
            avg_original_size = (
                self.original_size / self.successful_packets
            )
            avg_secured_size = (
                self.secured_size / self.successful_packets
            )
        else:
            avg_encryption = 0
            avg_decryption = 0
            avg_original_size = 0
            avg_secured_size = 0

        print("\n--- Simulation Metrics ---")
        print("Total packets:", self.total_packets)
        print("Successful packets:", self.successful_packets)
        print("Rejected packets:", self.rejected_packets)
        print(
            f"Average encryption time: "
            f"{avg_encryption:.8f} seconds"
        )
        print(
            f"Average decryption time: "
            f"{avg_decryption:.8f} seconds"
        )
        print(
            f"Average original packet size: "
            f"{avg_original_size:.2f} bytes"
        )
        print(
            f"Average secured packet size: "
            f"{avg_secured_size:.2f} bytes"
        )