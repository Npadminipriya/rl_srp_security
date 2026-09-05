from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from config import ProjectConfig
from node import Node


@dataclass(frozen=True)
class RoutingState:
    """
    Continuous RL-SRP state values for one candidate neighbour.
    """

    residual_energy: float
    link_quality: float
    trust_value: float
    distance_to_sink: float


@dataclass(frozen=True)
class DiscreteRoutingState:
    """
    Discretized state used later as a Q-table key.
    """

    energy_bin: int
    link_quality_bin: int
    trust_bin: int
    distance_bin: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (
            self.energy_bin,
            self.link_quality_bin,
            self.trust_bin,
            self.distance_bin,
        )


@dataclass(frozen=True)
class RewardComponents:
    """
    Individual normalized components of the routing reward.
    """

    trust: float
    distance_progress: float
    energy_cost: float
    delay: float
    packet_loss_rate: float
    total_reward: float


@dataclass(frozen=True)
class CandidateEvaluation:
    """
    Full RL evaluation of one possible next-hop action.
    """

    current_node_id: int
    candidate_node_id: int
    link_distance_m: float

    state: RoutingState
    discrete_state: DiscreteRoutingState
    reward: RewardComponents


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    """
    Restrict a numeric value to a specified interval.
    """

    return max(minimum, min(value, maximum))


def normalized_residual_energy(node: Node) -> float:
    """
    RE = remaining energy / initial energy.

    Sink energy is treated as fully available.
    """

    if node.node_type == "sink":
        return 1.0

    if node.initial_energy_j is None:
        return 1.0

    if node.residual_energy_j is None:
        return 1.0

    if node.initial_energy_j <= 0:
        return 0.0

    return clamp(
        node.residual_energy_j
        / node.initial_energy_j
    )


def calculate_link_quality(
    source: Node,
    candidate: Node,
    communication_range_m: float,
) -> float:
    """
    Initial distance-based link-quality approximation.

    A short communication link receives a higher score.
    A link at the exact communication-range boundary
    receives a score near zero.

    Later, this can be replaced or combined with the
    observed packet-success ratio.
    """

    distance = source.distance_to(candidate)

    if communication_range_m <= 0:
        raise ValueError(
            "Communication range must be positive."
        )

    return clamp(
        1.0 - (
            distance / communication_range_m
        )
    )


def maximum_possible_sink_distance(
    config: ProjectConfig,
) -> float:
    """
    Maximum geometric distance possible within
    the configured three-dimensional deployment region.
    """

    max_x_distance = max(
        config.sink.x_m,
        config.topology.area_x_m - config.sink.x_m,
    )

    max_y_distance = max(
        config.sink.y_m,
        config.topology.area_y_m - config.sink.y_m,
    )

    max_z_distance = config.topology.depth_m

    return sqrt(
        max_x_distance ** 2
        + max_y_distance ** 2
        + max_z_distance ** 2
    )


def normalized_distance_to_sink(
    node: Node,
    sink: Node,
    config: ProjectConfig,
) -> float:
    """
    DS = current distance to sink / maximum possible distance.

    Lower values mean that the candidate is closer to the sink.
    """

    maximum_distance = maximum_possible_sink_distance(
        config
    )

    if maximum_distance <= 0:
        return 0.0

    return clamp(
        node.distance_to_sink(sink)
        / maximum_distance
    )


def calculate_distance_progress(
    current_node: Node,
    candidate: Node,
    sink: Node,
    config: ProjectConfig,
) -> float:
    """
    Calculate normalized progress toward the sink.

    Positive:
        Candidate is closer to the sink.

    Zero:
        Candidate is at the same sink distance.

    Negative:
        Candidate moves temporarily farther away.
    """

    maximum_distance = maximum_possible_sink_distance(
        config
    )

    if maximum_distance <= 0:
        return 0.0

    current_distance = current_node.distance_to_sink(
        sink
    )

    candidate_distance = candidate.distance_to_sink(
        sink
    )

    progress = (
        current_distance - candidate_distance
    ) / maximum_distance

    return clamp(
        progress,
        minimum=-1.0,
        maximum=1.0,
    )


def estimate_energy_cost(
    source: Node,
    candidate: Node,
    config: ProjectConfig,
) -> float:
    """
    Estimate normalized transmission-energy cost.

    Transmission time:
        packet bits / data rate

    Basic energy:
        transmission power × transmission time

    A distance factor increases the relative cost
    for longer acoustic links.
    """

    packet_bits = (
        config.simulation.packet_size_bytes * 8
    )

    transmission_time = (
        packet_bits / config.energy.data_rate_bps
    )

    base_energy = (
        config.energy.transmit_power_w
        * transmission_time
    )

    distance = source.distance_to(candidate)

    distance_ratio = clamp(
        distance
        / config.topology.communication_range_m
    )

    estimated_energy = (
        base_energy
        * (1.0 + distance_ratio ** 2)
    )

    maximum_estimated_energy = (
        base_energy * 2.0
    )

    if maximum_estimated_energy <= 0:
        return 0.0

    return clamp(
        estimated_energy
        / maximum_estimated_energy
    )


def estimate_delay(
    source: Node,
    candidate: Node,
    config: ProjectConfig,
) -> float:
    """
    Estimate normalized one-hop acoustic delay.

    Acoustic propagation speed is approximated as
    1500 metres per second.

    The result is normalized against the delay at
    maximum communication range.
    """

    acoustic_speed_mps = 1500.0

    distance = source.distance_to(candidate)

    propagation_delay = (
        distance / acoustic_speed_mps
    )

    packet_bits = (
        config.simulation.packet_size_bytes * 8
    )

    transmission_delay = (
        packet_bits / config.energy.data_rate_bps
    )

    total_delay = (
        propagation_delay + transmission_delay
    )

    maximum_delay = (
        config.topology.communication_range_m
        / acoustic_speed_mps
        + transmission_delay
    )

    if maximum_delay <= 0:
        return 0.0

    return clamp(
        total_delay / maximum_delay
    )


def estimate_packet_loss_rate(
    link_quality: float,
) -> float:
    """
    For the initial environment, estimated PLR is
    the complement of link quality.

    Later, actual sent and received packet counts
    will replace this estimate.
    """

    return clamp(
        1.0 - link_quality
    )


def discretize_value(
    value: float,
    number_of_bins: int,
) -> int:
    """
    Convert a normalized value between 0 and 1
    into a discrete bin from 0 to bins - 1.
    """

    if number_of_bins < 2:
        raise ValueError(
            "Number of bins must be at least 2."
        )

    normalized = clamp(value)

    bin_index = int(
        normalized * number_of_bins
    )

    return min(
        bin_index,
        number_of_bins - 1,
    )


def discretize_state(
    state: RoutingState,
    config: ProjectConfig,
) -> DiscreteRoutingState:
    """
    Convert the continuous state vector into bins
    suitable for tabular Q-learning.
    """

    return DiscreteRoutingState(
        energy_bin=discretize_value(
            state.residual_energy,
            config.state.energy_bins,
        ),

        link_quality_bin=discretize_value(
            state.link_quality,
            config.state.link_quality_bins,
        ),

        trust_bin=discretize_value(
            state.trust_value,
            config.state.trust_bins,
        ),

        distance_bin=discretize_value(
            state.distance_to_sink,
            config.state.distance_bins,
        ),
    )


def calculate_reward(
    trust_value: float,
    distance_progress: float,
    energy_cost: float,
    delay: float,
    packet_loss_rate: float,
    config: ProjectConfig,
) -> RewardComponents:
    """
    Calculate a preview routing reward.

    Positive components:
        Trust
        Progress toward sink

    Negative components:
        Transmission energy
        Delay
        Packet loss
    """

    reward = (
        config.reward.trust_weight
        * trust_value

        + config.reward.distance_progress_weight
        * distance_progress

        - config.reward.energy_cost_weight
        * energy_cost

        - config.reward.delay_weight
        * delay

        - config.reward.packet_loss_weight
        * packet_loss_rate
    )

    return RewardComponents(
        trust=trust_value,
        distance_progress=distance_progress,
        energy_cost=energy_cost,
        delay=delay,
        packet_loss_rate=packet_loss_rate,
        total_reward=reward,
    )


def evaluate_candidate(
    current_node: Node,
    candidate: Node,
    sink: Node,
    trust_value: float,
    config: ProjectConfig,
) -> CandidateEvaluation:
    """
    Construct the complete RL state and preview reward
    for selecting one neighbour as the next hop.
    """

    link_distance = current_node.distance_to(
        candidate
    )

    residual_energy = normalized_residual_energy(
        candidate
    )

    link_quality = calculate_link_quality(
        source=current_node,
        candidate=candidate,
        communication_range_m=(
            config.topology.communication_range_m
        ),
    )

    distance_to_sink = normalized_distance_to_sink(
        node=candidate,
        sink=sink,
        config=config,
    )

    state = RoutingState(
        residual_energy=residual_energy,
        link_quality=link_quality,
        trust_value=clamp(trust_value),
        distance_to_sink=distance_to_sink,
    )

    discrete_state = discretize_state(
        state=state,
        config=config,
    )

    distance_progress = calculate_distance_progress(
        current_node=current_node,
        candidate=candidate,
        sink=sink,
        config=config,
    )

    energy_cost = estimate_energy_cost(
        source=current_node,
        candidate=candidate,
        config=config,
    )

    delay = estimate_delay(
        source=current_node,
        candidate=candidate,
        config=config,
    )

    packet_loss_rate = estimate_packet_loss_rate(
        link_quality
    )

    reward = calculate_reward(
        trust_value=trust_value,
        distance_progress=distance_progress,
        energy_cost=energy_cost,
        delay=delay,
        packet_loss_rate=packet_loss_rate,
        config=config,
    )

    return CandidateEvaluation(
        current_node_id=current_node.node_id,
        candidate_node_id=candidate.node_id,
        link_distance_m=link_distance,
        state=state,
        discrete_state=discrete_state,
        reward=reward,
    )