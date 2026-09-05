from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SimulationConfig:
    random_seed: int
    duration_seconds: int
    packet_size_bytes: int
    packet_generation_interval_seconds: float


@dataclass(frozen=True)
class TopologyConfig:
    sensor_nodes: int
    sink_nodes: int
    dedicated_relay_nodes: int

    area_x_m: float
    area_y_m: float
    depth_m: float
    minimum_sensor_depth_m: float

    communication_range_m: float
    minimum_neighbors: int
    maximum_generation_attempts: int

    deployment: str
    mobility_enabled: bool
    node_speed_mps: float


@dataclass(frozen=True)
class SinkConfig:
    placement: str
    x_m: float
    y_m: float
    z_m: float


@dataclass(frozen=True)
class EnergyConfig:
    initial_energy_j: float
    transmit_power_w: float
    receive_power_w: float
    data_rate_bps: float


@dataclass(frozen=True)
class TrustConfig:
    initial_value: float
    malicious_threshold: float
    memory_factor: float


@dataclass(frozen=True)
class QLearningConfig:
    learning_rate: float
    discount_factor: float

    epsilon_start: float
    epsilon_min: float
    epsilon_decay: float

    training_episodes: int
    maximum_hops_per_episode: int

    sink_reward: float
    dead_end_penalty: float
    loop_penalty: float


@dataclass(frozen=True)
class StateConfig:
    energy_bins: int
    link_quality_bins: int
    trust_bins: int
    distance_bins: int

@dataclass(frozen=True)
class RewardConfig:
    trust_weight: float
    distance_progress_weight: float
    energy_cost_weight: float
    delay_weight: float
    packet_loss_weight: float


@dataclass(frozen=True)
class SecurityConfig:
    aes_key_bits: int
    hmac_algorithm: str
    hmac_key_bytes: int
    replay_protection_enabled: bool


@dataclass(frozen=True)
class OutputConfig:
    directory: str
    save_topology_csv: bool
    save_packet_log_csv: bool
    save_metrics_csv: bool


@dataclass(frozen=True)
class ProjectConfig:
    simulation: SimulationConfig
    topology: TopologyConfig
    sink: SinkConfig
    energy: EnergyConfig
    trust: TrustConfig
    q_learning: QLearningConfig
    state: StateConfig
    reward: RewardConfig
    security: SecurityConfig
    output: OutputConfig


def require_range(
    name: str,
    value: float,
    minimum: float,
    maximum: float,
) -> None:
    """
    Check whether a numeric value lies within a valid range.
    """

    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}. "
            f"Received: {value}"
        )


def validate_config(config: ProjectConfig) -> None:
    """
    Validate all Phase 1 configuration values.
    """

    simulation = config.simulation
    topology = config.topology
    sink = config.sink
    energy = config.energy
    trust = config.trust
    q_learning = config.q_learning
    state = config.state
    reward = config.reward
    security = config.security

    # ----------------------------
    # Simulation validation
    # ----------------------------

    if simulation.duration_seconds <= 0:
        raise ValueError(
            "simulation.duration_seconds must be greater than zero."
        )

    if simulation.packet_size_bytes <= 0:
        raise ValueError(
            "simulation.packet_size_bytes must be greater than zero."
        )

    if simulation.packet_generation_interval_seconds <= 0:
        raise ValueError(
            "packet_generation_interval_seconds must be greater than zero."
        )

    # ----------------------------
    # Topology validation
    # ----------------------------

    if topology.sensor_nodes <= 0:
        raise ValueError(
            "topology.sensor_nodes must be greater than zero."
        )

    if topology.sink_nodes != 1:
        raise ValueError(
            "Phase 1 currently supports exactly one sink node."
        )

    if topology.dedicated_relay_nodes < 0:
        raise ValueError(
            "dedicated_relay_nodes cannot be negative."
        )

    if topology.area_x_m <= 0:
        raise ValueError("area_x_m must be positive.")

    if topology.area_y_m <= 0:
        raise ValueError("area_y_m must be positive.")

    if topology.depth_m <= 0:
        raise ValueError("depth_m must be positive.")

    if not 0 <= topology.minimum_sensor_depth_m < topology.depth_m:
        raise ValueError(
            "minimum_sensor_depth_m must lie inside the water volume."
        )

    if topology.communication_range_m <= 0:
        raise ValueError(
            "communication_range_m must be positive."
        )

    if topology.minimum_neighbors < 1:
        raise ValueError(
            "minimum_neighbors must be at least 1."
        )

    if topology.maximum_generation_attempts <= 0:
        raise ValueError(
            "maximum_generation_attempts must be positive."
        )

    if topology.deployment != "uniform_random_3d":
        raise ValueError(
            "Phase 1 supports only uniform_random_3d deployment."
        )

    # ----------------------------
    # Sink validation
    # ----------------------------

    if not 0 <= sink.x_m <= topology.area_x_m:
        raise ValueError(
            "The sink x-coordinate lies outside the deployment region."
        )

    if not 0 <= sink.y_m <= topology.area_y_m:
        raise ValueError(
            "The sink y-coordinate lies outside the deployment region."
        )

    if sink.z_m != 0:
        raise ValueError(
            "The surface sink must be placed at z = 0."
        )

    # ----------------------------
    # Energy validation
    # ----------------------------

    if energy.initial_energy_j <= 0:
        raise ValueError(
            "initial_energy_j must be greater than zero."
        )

    if energy.transmit_power_w <= 0:
        raise ValueError(
            "transmit_power_w must be greater than zero."
        )

    if energy.receive_power_w <= 0:
        raise ValueError(
            "receive_power_w must be greater than zero."
        )

    if energy.data_rate_bps <= 0:
        raise ValueError(
            "data_rate_bps must be greater than zero."
        )

    # ----------------------------
    # Trust validation
    # ----------------------------

    require_range(
        "trust.initial_value",
        trust.initial_value,
        0.0,
        1.0,
    )

    require_range(
        "trust.malicious_threshold",
        trust.malicious_threshold,
        0.0,
        1.0,
    )

    require_range(
        "trust.memory_factor",
        trust.memory_factor,
        0.0,
        1.0,
    )

    if trust.malicious_threshold >= trust.initial_value:
        raise ValueError(
            "The malicious threshold should be lower than "
            "the initial trust value."
        )

    # ----------------------------
    # Q-learning validation
    # ----------------------------

    require_range(
        "q_learning.learning_rate",
        q_learning.learning_rate,
        0.0,
        1.0,
    )

    require_range(
        "q_learning.discount_factor",
        q_learning.discount_factor,
        0.0,
        1.0,
    )

    require_range(
        "q_learning.epsilon_start",
        q_learning.epsilon_start,
        0.0,
        1.0,
    )

    require_range(
        "q_learning.epsilon_min",
        q_learning.epsilon_min,
        0.0,
        1.0,
    )

    require_range(
        "q_learning.epsilon_decay",
        q_learning.epsilon_decay,
        0.0,
        1.0,
    )

    if q_learning.epsilon_min > q_learning.epsilon_start:
        raise ValueError(
            "epsilon_min cannot be greater than epsilon_start."
        )
    if q_learning.training_episodes <= 0:
        raise ValueError(
            "training_episodes must be greater than zero."
    )

    if q_learning.maximum_hops_per_episode <= 0:
        raise ValueError(
            "maximum_hops_per_episode must be greater than zero."
    )

    if q_learning.sink_reward <= 0:
        raise ValueError(
            "sink_reward must be greater than zero."
    )

    if q_learning.dead_end_penalty >= 0:
        raise ValueError(
                "dead_end_penalty must be negative."
    )

    if q_learning.loop_penalty >= 0:
        raise ValueError(
            "loop_penalty must be negative."
    )

    # ----------------------------
    # State validation
    # ----------------------------

    state_bins = {
        "energy_bins": state.energy_bins,
        "link_quality_bins": state.link_quality_bins,
        "trust_bins": state.trust_bins,
        "distance_bins": state.distance_bins,
    }

    for name, value in state_bins.items():
        if value < 2:
            raise ValueError(
                f"state.{name} must be at least 2."
            )

    # ----------------------------
    # Reward validation
    # ----------------------------

    reward_weights = {
        "trust_weight": reward.trust_weight,
        "distance_progress_weight": (
            reward.distance_progress_weight
        ),
        "energy_cost_weight": reward.energy_cost_weight,
        "delay_weight": reward.delay_weight,
        "packet_loss_weight": reward.packet_loss_weight,
    }   

    for name, value in reward_weights.items():
        if value < 0:
            raise ValueError(
                f"reward.{name} cannot be negative."
            )

    weight_sum = sum(reward_weights.values())

    if abs(weight_sum - 1.0) > 1e-9:
        raise ValueError(
            "All reward weights must add up to 1.0. "
            f"Current sum: {weight_sum}"
        )

    # ----------------------------
    # Security validation
    # ----------------------------

    if security.aes_key_bits not in (128, 192, 256):
        raise ValueError(
            "AES key size must be 128, 192, or 256 bits."
        )

    if security.hmac_algorithm.upper() != "SHA-256":
        raise ValueError(
            "This implementation currently supports HMAC-SHA-256."
        )

    if security.hmac_key_bytes < 32:
        raise ValueError(
            "The HMAC key should contain at least 32 bytes."
        )


def load_config(
    path: str | Path | None = None,
) -> ProjectConfig:
    """
    Load config.json and convert it into typed configuration objects.
    """

    if path is None:
        config_path = Path(__file__).with_name("config.json")
    else:
        config_path = Path(path)

    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Configuration file was not found: {config_path}"
        ) from error

    try:
        raw: dict[str, Any] = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in configuration file: {error}"
        ) from error

    try:
        config = ProjectConfig(
            simulation=SimulationConfig(
                **raw["simulation"]
            ),
            topology=TopologyConfig(
                **raw["topology"]
            ),
            sink=SinkConfig(
                **raw["sink"]
            ),
            energy=EnergyConfig(
                **raw["energy"]
            ),
            trust=TrustConfig(
                **raw["trust"]
            ),
            q_learning=QLearningConfig(
                **raw["q_learning"]
            ),
            state=StateConfig(
                **raw["state"]
            ),
            reward=RewardConfig(
                **raw["reward"]
            ),
            security=SecurityConfig(
                **raw["security"]
            ),
            output=OutputConfig(
                **raw["output"]
            ),
        )

    except KeyError as error:
        raise ValueError(
            f"A required configuration section is missing: {error}"
        ) from error

    except TypeError as error:
        raise ValueError(
            f"One or more configuration fields are invalid: {error}"
        ) from error

    validate_config(config)

    return config


def print_configuration(config: ProjectConfig) -> None:
    """
    Display a readable summary after successful validation.
    """

    total_devices = (
        config.topology.sensor_nodes
        + config.topology.sink_nodes
        + config.topology.dedicated_relay_nodes
    )

    estimated_packets_per_sensor = int(
        config.simulation.duration_seconds
        / config.simulation.packet_generation_interval_seconds
    )

    print("=" * 58)
    print("RL-SRP PHASE 1: CONFIGURATION")
    print("=" * 58)

    print("\nSimulation")
    print(f"  Random seed             : {config.simulation.random_seed}")
    print(
        f"  Simulation duration     : "
        f"{config.simulation.duration_seconds} seconds"
    )
    print(
        f"  Packet size             : "
        f"{config.simulation.packet_size_bytes} bytes"
    )
    print(
        f"  Packet interval         : "
        f"{config.simulation.packet_generation_interval_seconds} seconds"
    )
    print(
        f"  Packets per sensor      : "
        f"{estimated_packets_per_sensor}"
    )

    print("\nTopology")
    print(
        f"  Sensor nodes            : "
        f"{config.topology.sensor_nodes}"
    )
    print(
        f"  Sink nodes              : "
        f"{config.topology.sink_nodes}"
    )
    print(
        f"  Dedicated relay nodes   : "
        f"{config.topology.dedicated_relay_nodes}"
    )
    print(f"  Total devices           : {total_devices}")
    print(
        f"  Deployment region       : "
        f"{config.topology.area_x_m} × "
        f"{config.topology.area_y_m} × "
        f"{config.topology.depth_m} m"
    )
    print(
        f"  Communication range     : "
        f"{config.topology.communication_range_m} m"
    )
    print(
        f"  Minimum neighbours      : "
        f"{config.topology.minimum_neighbors}"
    )
    print(
        f"  Mobility enabled        : "
        f"{config.topology.mobility_enabled}"
    )

    print("\nSink")
    print(
        f"  Position                : "
        f"({config.sink.x_m}, "
        f"{config.sink.y_m}, "
        f"{config.sink.z_m})"
    )

    print("\nEnergy")
    print(
        f"  Initial node energy     : "
        f"{config.energy.initial_energy_j} J"
    )
    print(
        f"  Transmission power      : "
        f"{config.energy.transmit_power_w} W"
    )
    print(
        f"  Reception power         : "
        f"{config.energy.receive_power_w} W"
    )

    print("\nTrust")
    print(
        f"  Initial trust           : "
        f"{config.trust.initial_value}"
    )
    print(
        f"  Malicious threshold     : "
        f"{config.trust.malicious_threshold}"
    )
    print(
        f"  Memory factor           : "
        f"{config.trust.memory_factor}"
    )

    print("\nQ-learning")

    print(
        f"  Learning rate           : "
        f"{config.q_learning.learning_rate}"
    )

    print(
        f"  Discount factor         : "
        f"{config.q_learning.discount_factor}"
    )

    print(
        f"  Initial epsilon         : "
        f"{config.q_learning.epsilon_start}"
    )

    print(
        f"  Minimum epsilon         : "
        f"{config.q_learning.epsilon_min}"
    )

    print(
        f"  Epsilon decay           : "
        f"{config.q_learning.epsilon_decay}"
    )

    print(
        f"  Training episodes       : "
        f"{config.q_learning.training_episodes}"
    )

    print(
        f"  Maximum hops/episode    : "
        f"{config.q_learning.maximum_hops_per_episode}"
    )

    print(
        f"  Sink reward             : "
        f"{config.q_learning.sink_reward}"
    )

    print(
        f"  Dead-end penalty        : "
        f"{config.q_learning.dead_end_penalty}"
    )

    print(
        f"  Loop penalty            : "
        f"{config.q_learning.loop_penalty}"
    )

    print("\nSecurity")
    print(
        f"  AES key length          : "
        f"{config.security.aes_key_bits} bits"
    )
    print(
        f"  HMAC algorithm          : "
        f"{config.security.hmac_algorithm}"
    )
    print(
        f"  Replay protection       : "
        f"{config.security.replay_protection_enabled}"
    )

    print("\nConfiguration validation: PASSED")
    print("Phase 1 completed successfully.")
    print("=" * 58)


def main() -> None:
    try:
        config = load_config()
        print_configuration(config)

    except (ValueError, FileNotFoundError) as error:
        print("=" * 58)
        print("CONFIGURATION ERROR")
        print("=" * 58)
        print(error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()