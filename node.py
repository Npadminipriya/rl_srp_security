from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt


@dataclass
class Node:
    """
    Represents either an underwater sensor node
    or the surface sink node.
    """

    node_id: int
    node_type: str

    x: float
    y: float
    z: float

    initial_energy_j: float | None
    residual_energy_j: float | None

    neighbours: list[int] = field(default_factory=list)

    def distance_to(self, other: "Node") -> float:
        """
        Calculate the 3D Euclidean distance
        between this node and another node.
        """

        return sqrt(
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        )

    def distance_to_sink(self, sink: "Node") -> float:
        """
        Calculate the distance from this node
        to the surface sink.
        """

        return self.distance_to(sink)