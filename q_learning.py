from __future__ import annotations

import random

from collections import defaultdict
from dataclasses import dataclass

from config import ProjectConfig
from node import Node
from rl_state import (
    CandidateEvaluation,
    evaluate_candidate,
)
from topology import TopologyResult
from trust_manager import TrustManager


# ============================================================
# Q-TABLE KEY
# ============================================================

QKey = tuple[
    int,
    tuple[int, int, int, int],
    int,
]


# ============================================================
# TRAINING RESULT
# ============================================================

@dataclass
class TrainingEpisodeResult:
    episode: int
    source_id: int
    delivered: bool
    hops: int
    total_reward: float
    epsilon: float


# ============================================================
# ROUTING RESULT
# ============================================================

@dataclass
class RLRoutingResult:
    packet_id: int
    source_id: int

    delivered: bool
    route: list[int]
    hop_count: int

    total_reward: float
    reason: str


# ============================================================
# Q-LEARNING ROUTER
# ============================================================

class QLearningRouter:
    """
    Tabular Q-learning router for RL-SRP.

    Phase 5:
        Uses fixed initial trust.

    Phase 6:
        Uses TrustManager for dynamic trust
        and blacklist enforcement.
    """

    def __init__(
        self,
        topology: TopologyResult,
        config: ProjectConfig,
        trust_manager: TrustManager | None = None,
    ) -> None:

        self.topology = topology
        self.config = config

        # Optional Phase-6 trust manager
        self.trust_manager = trust_manager

        # Quick node lookup
        self.node_by_id: dict[int, Node] = {
            node.node_id: node
            for node in topology.nodes
        }

        # Sink is always Node 0
        self.sink = self.node_by_id[0]

        # Q-values start at 0
        self.q_table: defaultdict[
            QKey,
            float,
        ] = defaultdict(float)

        # Reproducible random generator
        self.rng = random.Random(
            config.simulation.random_seed
        )

        # Initial exploration probability
        self.epsilon = (
            config.q_learning.epsilon_start
        )

    # ========================================================
    # CANDIDATE ACTION EVALUATION
    # ========================================================

    def evaluate_actions(
        self,
        current_node: Node,
        visited: set[int],
    ) -> list[CandidateEvaluation]:
        """
        Evaluate every currently usable neighbour.

        Rules:
        1. Do not revisit already visited nodes.
        2. If TrustManager exists, exclude blacklisted nodes.
        3. If TrustManager exists, use dynamic trust.
        4. Otherwise use initial trust = 0.5.

        Unlike Phase 3 greedy routing, a neighbour is NOT
        required to be closer to the sink.

        This allows RL to escape local minima.
        """

        evaluations: list[
            CandidateEvaluation
        ] = []

        for neighbour_id in current_node.neighbours:

            # --------------------------------------------
            # Avoid loops
            # --------------------------------------------

            if neighbour_id in visited:
                continue

            # --------------------------------------------
            # Blacklist enforcement
            # --------------------------------------------

            if (
                self.trust_manager is not None
                and self.trust_manager.is_blacklisted(
                    current_node.node_id,
                    neighbour_id,
                )
            ):
                continue

            candidate = self.node_by_id[
                neighbour_id
            ]

            # --------------------------------------------
            # Trust value
            # --------------------------------------------

            if self.trust_manager is None:

                # Phase 5:
                # fixed neutral trust
                trust_value = (
                    self.config.trust.initial_value
                )

            else:

                # Phase 6:
                # dynamically maintained trust
                trust_value = (
                    self.trust_manager.get_trust(
                        current_node.node_id,
                        neighbour_id,
                    )
                )

            # --------------------------------------------
            # Build state + reward
            # --------------------------------------------

            evaluation = evaluate_candidate(
                current_node=current_node,
                candidate=candidate,
                sink=self.sink,
                trust_value=trust_value,
                config=self.config,
            )

            evaluations.append(
                evaluation
            )

        return evaluations

    # ========================================================
    # Q-TABLE KEY
    # ========================================================

    def make_q_key(
        self,
        evaluation: CandidateEvaluation,
    ) -> QKey:
        """
        Q-table key:

        (
            current node,
            discretized state,
            candidate neighbour
        )
        """

        return (
            evaluation.current_node_id,
            evaluation.discrete_state.as_tuple(),
            evaluation.candidate_node_id,
        )

    # ========================================================
    # EPSILON-GREEDY ACTION SELECTION
    # ========================================================

    def choose_action(
        self,
        evaluations: list[CandidateEvaluation],
        training: bool = True,
    ) -> CandidateEvaluation | None:
        """
        During training:

            epsilon probability
                -> random exploration

            1 - epsilon probability
                -> best-known Q-value

        During testing:
            always use the highest Q-value.
        """

        if not evaluations:
            return None

        # --------------------------------------------
        # Exploration
        # --------------------------------------------

        if (
            training
            and self.rng.random() < self.epsilon
        ):
            return self.rng.choice(
                evaluations
            )

        # --------------------------------------------
        # Exploitation
        # --------------------------------------------

        best_q = max(
            self.q_table[
                self.make_q_key(evaluation)
            ]
            for evaluation in evaluations
        )

        # Multiple actions can have same Q-value
        best_actions = [
            evaluation
            for evaluation in evaluations
            if self.q_table[
                self.make_q_key(evaluation)
            ] == best_q
        ]

        return self.rng.choice(
            best_actions
        )

    # ========================================================
    # MAXIMUM FUTURE Q
    # ========================================================

    def maximum_future_q(
        self,
        current_node: Node,
        visited: set[int],
    ) -> float:
        """
        Find maximum Q-value available
        from the next state.
        """

        future_actions = self.evaluate_actions(
            current_node=current_node,
            visited=visited,
        )

        if not future_actions:
            return 0.0

        return max(
            self.q_table[
                self.make_q_key(action)
            ]
            for action in future_actions
        )

    # ========================================================
    # Q-VALUE UPDATE
    # ========================================================

    def update_q_value(
        self,
        evaluation: CandidateEvaluation,
        reward: float,
        next_node: Node,
        visited: set[int],
        terminal: bool,
    ) -> None:
        """
        Standard Q-learning update:

        Q(s,a) =
            Q(s,a)
            + alpha *
              [
                reward
                + gamma * max Q(s',a')
                - Q(s,a)
              ]
        """

        key = self.make_q_key(
            evaluation
        )

        old_q = self.q_table[key]

        # --------------------------------------------
        # Future value
        # --------------------------------------------

        if terminal:
            future_q = 0.0

        else:
            future_q = self.maximum_future_q(
                current_node=next_node,
                visited=visited,
            )

        learning_rate = (
            self.config.q_learning.learning_rate
        )

        discount_factor = (
            self.config.q_learning.discount_factor
        )

        target = (
            reward
            + discount_factor * future_q
        )

        new_q = (
            old_q
            + learning_rate
            * (
                target
                - old_q
            )
        )

        self.q_table[key] = new_q

    # ========================================================
    # TRAIN ONE EPISODE
    # ========================================================

    def train_episode(
        self,
        episode_number: int,
        source_id: int,
    ) -> TrainingEpisodeResult:
        """
        Train the Q-learning router for
        one source-to-sink episode.
        """

        current_node = self.node_by_id[
            source_id
        ]

        visited = {
            current_node.node_id
        }

        total_reward = 0.0
        hops = 0
        delivered = False

        maximum_hops = (
            self.config.q_learning
            .maximum_hops_per_episode
        )

        while hops < maximum_hops:

            # --------------------------------------------
            # Already reached sink
            # --------------------------------------------

            if current_node.node_id == 0:

                delivered = True
                break

            # --------------------------------------------
            # Find valid neighbour actions
            # --------------------------------------------

            actions = self.evaluate_actions(
                current_node=current_node,
                visited=visited,
            )

            # --------------------------------------------
            # Dead end
            # --------------------------------------------

            if not actions:

                total_reward += (
                    self.config.q_learning
                    .dead_end_penalty
                )

                break

            # --------------------------------------------
            # Select action
            # --------------------------------------------

            action = self.choose_action(
                evaluations=actions,
                training=True,
            )

            if action is None:
                break

            next_node = self.node_by_id[
                action.candidate_node_id
            ]

            # --------------------------------------------
            # Phase-6 trust feedback
            # --------------------------------------------
            #
            # For now, a completed normal hop is treated
            # as successful forwarding.
            #
            # Later attack phases can call:
            #
            # success=False
            #
            # for blackhole/selective-forwarding/etc.
            # --------------------------------------------

            if self.trust_manager is not None:

                self.trust_manager.record_forwarding(
                    observer_id=current_node.node_id,
                    neighbour_id=next_node.node_id,
                    success=True,
                )

            # --------------------------------------------
            # Base reward
            # --------------------------------------------

            reward = (
                action.reward.total_reward
            )

            # --------------------------------------------
            # Sink reached?
            # --------------------------------------------

            terminal = (
                next_node.node_id == 0
            )

            if terminal:

                reward += (
                    self.config.q_learning
                    .sink_reward
                )

            # --------------------------------------------
            # Q-learning update
            # --------------------------------------------

            self.update_q_value(
                evaluation=action,
                reward=reward,
                next_node=next_node,
                visited=visited,
                terminal=terminal,
            )

            total_reward += reward

            # --------------------------------------------
            # Move packet
            # --------------------------------------------

            current_node = next_node

            visited.add(
                current_node.node_id
            )

            hops += 1

            if terminal:

                delivered = True
                break

        # --------------------------------------------
        # Maximum-hop penalty
        # --------------------------------------------

        if (
            not delivered
            and hops >= maximum_hops
        ):

            total_reward += (
                self.config.q_learning
                .loop_penalty
            )

        return TrainingEpisodeResult(
            episode=episode_number,
            source_id=source_id,
            delivered=delivered,
            hops=hops,
            total_reward=total_reward,
            epsilon=self.epsilon,
        )

    # ========================================================
    # EPSILON DECAY
    # ========================================================

    def decay_epsilon(
        self,
    ) -> None:
        """
        Gradually reduce exploration.

        epsilon cannot fall below epsilon_min.
        """

        self.epsilon = max(
            self.config.q_learning.epsilon_min,

            self.epsilon
            * self.config.q_learning.epsilon_decay,
        )

    # ========================================================
    # FULL TRAINING
    # ========================================================

    def train(
        self,
    ) -> list[TrainingEpisodeResult]:
        """
        Train the router for the configured
        number of episodes.
        """

        results: list[
            TrainingEpisodeResult
        ] = []

        sensor_ids = [
            node.node_id
            for node in self.topology.nodes
            if node.node_type == "sensor"
        ]

        episodes = (
            self.config.q_learning
            .training_episodes
        )

        for episode in range(
            1,
            episodes + 1,
        ):

            # Choose random source sensor
            source_id = self.rng.choice(
                sensor_ids
            )

            result = self.train_episode(
                episode_number=episode,
                source_id=source_id,
            )

            results.append(
                result
            )

            self.decay_epsilon()

        return results

    # ========================================================
    # ROUTE PACKET AFTER TRAINING
    # ========================================================

    def route_packet(
        self,
        packet_id: int,
        source_id: int,
    ) -> RLRoutingResult:
        """
        Route one packet using the learned policy.

        No exploration is used during testing.
        """

        current_node = self.node_by_id[
            source_id
        ]

        route = [
            current_node.node_id
        ]

        visited = {
            current_node.node_id
        }

        total_reward = 0.0

        maximum_hops = (
            self.config.q_learning
            .maximum_hops_per_episode
        )

        for _ in range(
            maximum_hops
        ):

            # --------------------------------------------
            # Sink reached
            # --------------------------------------------

            if current_node.node_id == 0:

                return RLRoutingResult(
                    packet_id=packet_id,
                    source_id=source_id,
                    delivered=True,
                    route=route,
                    hop_count=len(route) - 1,
                    total_reward=total_reward,
                    reason="packet delivered",
                )

            # --------------------------------------------
            # Available neighbours
            # --------------------------------------------

            actions = self.evaluate_actions(
                current_node=current_node,
                visited=visited,
            )

            if not actions:

                return RLRoutingResult(
                    packet_id=packet_id,
                    source_id=source_id,
                    delivered=False,
                    route=route,
                    hop_count=len(route) - 1,
                    total_reward=total_reward,
                    reason="no available next hop",
                )

            # --------------------------------------------
            # Exploit learned Q-values
            # --------------------------------------------

            action = self.choose_action(
                evaluations=actions,
                training=False,
            )

            if action is None:

                return RLRoutingResult(
                    packet_id=packet_id,
                    source_id=source_id,
                    delivered=False,
                    route=route,
                    hop_count=len(route) - 1,
                    total_reward=total_reward,
                    reason="no action selected",
                )

            next_node = self.node_by_id[
                action.candidate_node_id
            ]

            reward = (
                action.reward.total_reward
            )

            # --------------------------------------------
            # Sink terminal reward
            # --------------------------------------------

            if next_node.node_id == 0:

                reward += (
                    self.config.q_learning
                    .sink_reward
                )

            total_reward += reward

            # --------------------------------------------
            # Move packet
            # --------------------------------------------

            current_node = next_node

            route.append(
                current_node.node_id
            )

            visited.add(
                current_node.node_id
            )

            # --------------------------------------------
            # Immediate sink delivery
            # --------------------------------------------

            if current_node.node_id == 0:

                return RLRoutingResult(
                    packet_id=packet_id,
                    source_id=source_id,
                    delivered=True,
                    route=route,
                    hop_count=len(route) - 1,
                    total_reward=total_reward,
                    reason="packet delivered",
                )

        # --------------------------------------------
        # Maximum-hop failure
        # --------------------------------------------

        return RLRoutingResult(
            packet_id=packet_id,
            source_id=source_id,
            delivered=False,
            route=route,
            hop_count=len(route) - 1,
            total_reward=total_reward,
            reason="maximum hop limit reached",
        )