from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# RL-SRP RESEARCH GRAPH GENERATION
# ============================================================
#
# Generates:
#
#   Routing / RL evaluation
#       Fig 1  - Packet Delivery Ratio
#       Fig 2  - Packet Loss
#       Fig 3  - Average Hop Count
#       Fig 4  - Hop Count Per Packet
#       Fig 5  - Q-Learning Reward Convergence
#       Fig 6  - Q-Learning Training Success Rate
#       Fig 7  - Epsilon Decay
#
#   Security evaluation
#       Fig 8  - Attack Detection Performance
#       Fig 9  - Packet Delivery Under Attacks
#       Fig 10 - Packet Drop Ratio Under Attacks
#       Fig 11 - Trust Degradation
#       Fig 12 - Blacklist Activation
#       Fig 13 - Attack Impact Comparison
#       Fig 14 - Route Failures Under Attacks
#       Fig 15 - Security vs Routing Cost
#       Fig 16 - Overall Security Metrics
#
# IMPORTANT:
# Security figures are generated from the actual Phase 8-11
# CSV files.
#
# No attack PDR/drop/trust values are hardcoded.
#
# ============================================================


# ============================================================
# PATH CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("outputs")

FIGURE_DIR = (
    OUTPUT_DIR
    / "research_figures"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# ROUTING INPUT FILES
# ============================================================

GREEDY_FILE = (
    OUTPUT_DIR
    / "packet_forwarding_results.csv"
)

QLEARNING_FILE = (
    OUTPUT_DIR
    / "rl_packet_routing_results.csv"
)

TRAINING_FILE = (
    OUTPUT_DIR
    / "q_learning_training.csv"
)


# ============================================================
# SECURITY INPUT FILES
# ============================================================

ATTACK_FILES = {
    "Blackhole": (
        OUTPUT_DIR
        / "blackhole_attack_results.csv"
    ),
    "Selective Forwarding": (
        OUTPUT_DIR
        / "selective_forwarding_results.csv"
    ),
    "Replay": (
        OUTPUT_DIR
        / "replay_attack_results.csv"
    ),
    "Modification": (
        OUTPUT_DIR
        / "packet_modification_results.csv"
    ),
}


# ============================================================
# GRAPH SETTINGS
# ============================================================

FIGURE_DPI = 300

MOVING_AVERAGE_WINDOW = 100


# ============================================================
# GENERAL HELPERS
# ============================================================

def require_file(path: Path) -> None:
    """
    Check that an input CSV exists.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file was not found:\n"
            f"  {path}\n"
            f"\nRun the corresponding simulation phase first."
        )


def save_graph(
    figure,
    filename: str,
) -> None:
    """
    Save a graph as a high-resolution PNG.
    """

    output_path = (
        FIGURE_DIR
        / filename
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def to_bool(value) -> bool:
    """
    Convert common CSV representations of boolean values
    into a Python boolean.
    """

    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    if isinstance(value, (int, float)):

        return bool(value)

    text = (
        str(value)
        .strip()
        .lower()
    )

    return text in {
        "true",
        "1",
        "yes",
        "y",
        "delivered",
        "success",
        "successful",
    }


def safe_numeric(
    series: pd.Series,
) -> pd.Series:
    """
    Convert a pandas series to numeric values.
    Invalid values become NaN.
    """

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def moving_average(
    values: pd.Series,
    window: int,
) -> pd.Series:
    """
    Calculate a rolling moving average.
    """

    return (
        values
        .rolling(
            window=window,
            min_periods=1,
        )
        .mean()
    )


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    description: str,
) -> str:
    """
    Find a column using several possible names.

    This prevents graph_gen.py from depending on one exact
    column naming convention.
    """

    normalized_columns = {
        str(column)
        .strip()
        .lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        key = (
            candidate
            .strip()
            .lower()
        )

        if key in normalized_columns:
            return normalized_columns[key]

    raise KeyError(
        "\nCould not find "
        f"{description}.\n\n"
        f"Expected one of:\n"
        f"  {candidates}\n\n"
        f"Actual CSV columns are:\n"
        f"  {list(df.columns)}"
    )


# ============================================================
# ROUTING CSV NORMALIZATION
# ============================================================

def normalize_routing_dataframe(
    df: pd.DataFrame,
    filename: str,
) -> pd.DataFrame:
    """
    Normalize the existing routing CSV into the common
    columns required by the graph generator.

    The project may use names such as:

        packet_id
        packet
        id

    for packet IDs, and:

        delivered
        packet_delivered
        success

    for delivery status.

    This function creates the standard internal names:

        packet_id
        packet_delivered
        hop_count
    """

    result = df.copy()

    # --------------------------------------------------------
    # Print columns once so schema problems are easy to debug.
    # --------------------------------------------------------

    print(
        f"\nLoaded {filename}"
    )

    print(
        f"Columns: {list(result.columns)}"
    )

    # --------------------------------------------------------
    # Packet ID
    # --------------------------------------------------------

    packet_column = find_column(
        result,
        [
            "packet_id",
            "packet",
            "id",
            "packet_number",
        ],
        f"packet ID column in {filename}",
    )

    result["packet_id"] = safe_numeric(
        result[packet_column]
    )

    # --------------------------------------------------------
    # Delivery status
    # --------------------------------------------------------

    delivery_column = find_column(
        result,
        [
            "packet_delivered",
            "delivered",
            "delivery",
            "success",
            "successful",
            "packet_delivery",
            "is_delivered",
            "delivery_success",
        ],
        f"packet delivery column in {filename}",
    )

    result["packet_delivered"] = (
        result[delivery_column]
        .apply(to_bool)
    )

    # --------------------------------------------------------
    # Hop count
    # --------------------------------------------------------

    hop_column = find_column(
        result,
        [
            "hop_count",
            "hops",
            "hop",
            "route_hops",
            "number_of_hops",
        ],
        f"hop-count column in {filename}",
    )

    result["hop_count"] = safe_numeric(
        result[hop_column]
    )

    return result


# ============================================================
# LOAD ROUTING DATA
# ============================================================

def load_routing_data():
    """
    Load the existing routing and training CSV files.
    """

    require_file(
        GREEDY_FILE
    )

    require_file(
        QLEARNING_FILE
    )

    require_file(
        TRAINING_FILE
    )

    greedy_raw = pd.read_csv(
        GREEDY_FILE
    )

    qlearning_raw = pd.read_csv(
        QLEARNING_FILE
    )

    training = pd.read_csv(
        TRAINING_FILE
    )

    greedy = normalize_routing_dataframe(
        greedy_raw,
        "packet_forwarding_results.csv",
    )

    qlearning = normalize_routing_dataframe(
        qlearning_raw,
        "rl_packet_routing_results.csv",
    )

    return (
        greedy,
        qlearning,
        training,
    )


# ============================================================
# SECURITY CSV LOADING
# ============================================================

def load_attack_results():
    """
    Load all four Phase 8-11 attack result files.

    Each file is normalized into common internal fields.
    """

    attack_data = {}

    for attack_name, path in ATTACK_FILES.items():

        require_file(
            path
        )

        df = pd.read_csv(
            path
        )

        if df.empty:

            raise ValueError(
                f"\nAttack result file is empty:\n"
                f"  {path}"
            )

        print(
            f"\nLoaded {path.name}"
        )

        print(
            f"Columns: {list(df.columns)}"
        )

        # ----------------------------------------------------
        # Packet delivered
        # ----------------------------------------------------

        if "packet_delivered" in df.columns:

            df["packet_delivered_bool"] = (
                df["packet_delivered"]
                .apply(to_bool)
            )

        elif "delivered" in df.columns:

            df["packet_delivered_bool"] = (
                df["delivered"]
                .apply(to_bool)
            )

        else:

            # Fall back to result/status.
            if "result" in df.columns:

                result_text = (
                    df["result"]
                    .astype(str)
                    .str.upper()
                )

                df["packet_delivered_bool"] = (
                    result_text
                    == "DELIVERED"
                )

            else:

                df["packet_delivered_bool"] = False

        # ----------------------------------------------------
        # Attack drop column
        # ----------------------------------------------------

        drop_columns = [
            column
            for column in df.columns
            if str(column)
            .lower()
            .startswith(
                "dropped_by_"
            )
        ]

        if drop_columns:

            drop_column = (
                drop_columns[0]
            )

            df["attack_dropped"] = (
                df[drop_column]
                .apply(to_bool)
            )

        elif "dropped" in df.columns:

            df["attack_dropped"] = (
                df["dropped"]
                .apply(to_bool)
            )

        elif "result" in df.columns:

            result_text = (
                df["result"]
                .astype(str)
                .str.upper()
            )

            df["attack_dropped"] = (
                result_text
                .str.contains(
                    "DROP|DETECTED",
                    regex=True,
                    na=False,
                )
            )

        else:

            df["attack_dropped"] = False

        # ----------------------------------------------------
        # Route-found field
        # ----------------------------------------------------

        if "rl_route_found" in df.columns:

            df["route_found_bool"] = (
                df["rl_route_found"]
                .apply(to_bool)
            )

        elif "route_found" in df.columns:

            df["route_found_bool"] = (
                df["route_found"]
                .apply(to_bool)
            )

        else:

            df["route_found_bool"] = True

        # ----------------------------------------------------
        # Blacklist field
        # ----------------------------------------------------

        if "blacklisted" in df.columns:

            df["blacklisted_bool"] = (
                df["blacklisted"]
                .apply(to_bool)
            )

        else:

            df["blacklisted_bool"] = False

        # ----------------------------------------------------
        # Numeric fields
        # ----------------------------------------------------

        for column in [
            "packet_id",
            "source_id",
            "hop_count",
            "trust_before",
            "trust_after",
        ]:

            if column in df.columns:

                df[column] = safe_numeric(
                    df[column]
                )

        attack_data[
            attack_name
        ] = df

    return attack_data


# ============================================================
# SECURITY METRIC CALCULATION
# ============================================================

def calculate_attack_metrics(
    attack_data,
) -> pd.DataFrame:
    """
    Calculate security metrics directly from the Phase 8-11
    result CSVs.
    """

    rows = []

    for attack_name, df in attack_data.items():

        total_packets = len(
            df
        )

        delivered_packets = int(
            df[
                "packet_delivered_bool"
            ].sum()
        )

        dropped_packets = int(
            df[
                "attack_dropped"
            ].sum()
        )

        # ----------------------------------------------------
        # A route failure is a packet that was not delivered
        # and was not dropped by the attack mechanism.
        # ----------------------------------------------------

        route_failures = int(
            (
                ~df[
                    "packet_delivered_bool"
                ]
                &
                ~df[
                    "attack_dropped"
                ]
            ).sum()
        )

        blacklist_events = int(
            df[
                "blacklisted_bool"
            ].sum()
        )

        # ----------------------------------------------------
        # Detection is considered observed when:
        #
        #   - attack-related drop/detection events exist, OR
        #   - blacklist activation occurred.
        #
        # This reflects the current Phase 8-11 implementation.
        # ----------------------------------------------------

        detected = (
            dropped_packets > 0
            or blacklist_events > 0
        )

        if total_packets > 0:

            pdr = (
                delivered_packets
                / total_packets
                * 100
            )

            drop_ratio = (
                dropped_packets
                / total_packets
                * 100
            )

        else:

            pdr = 0.0
            drop_ratio = 0.0

        # ----------------------------------------------------
        # Trust
        # ----------------------------------------------------

        if "trust_after" in df.columns:

            trust_values = (
                df[
                    "trust_after"
                ]
                .dropna()
            )

        else:

            trust_values = pd.Series(
                dtype=float
            )

        if len(
            trust_values
        ) > 0:

            minimum_trust = float(
                trust_values.min()
            )

            maximum_trust = float(
                trust_values.max()
            )

        else:

            minimum_trust = math.nan
            maximum_trust = math.nan

        # ----------------------------------------------------
        # Trust degradation relative to initial trust 1.0.
        # ----------------------------------------------------

        if not math.isnan(
            minimum_trust
        ):

            trust_degradation = (
                1.0
                - minimum_trust
            ) * 100

        else:

            trust_degradation = math.nan

        # ----------------------------------------------------
        # Average hop count
        # ----------------------------------------------------

        if "hop_count" in df.columns:

            hop_values = (
                df[
                    "hop_count"
                ]
                .dropna()
            )

            if len(
                hop_values
            ) > 0:

                average_hops = float(
                    hop_values.mean()
                )

            else:

                average_hops = math.nan

        else:

            average_hops = math.nan

        rows.append(
            {
                "attack": attack_name,
                "total_packets": total_packets,
                "delivered_packets": delivered_packets,
                "dropped_packets": dropped_packets,
                "route_failures": route_failures,
                "pdr": pdr,
                "drop_ratio": drop_ratio,
                "attack_detected": (
                    100.0
                    if detected
                    else 0.0
                ),
                "blacklist_events": (
                    blacklist_events
                ),
                "minimum_trust": (
                    minimum_trust
                ),
                "maximum_trust": (
                    maximum_trust
                ),
                "trust_degradation": (
                    trust_degradation
                ),
                "average_hops": (
                    average_hops
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# FIGURE 1
# PACKET DELIVERY RATIO
# ============================================================

def figure_1_pdr(
    greedy,
    qlearning,
):

    greedy_pdr = (
        greedy[
            "packet_delivered"
        ]
        .mean()
        * 100
    )

    qlearning_pdr = (
        qlearning[
            "packet_delivered"
        ]
        .mean()
        * 100
    )

    figure = plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        [
            "Greedy",
            "Q-learning",
        ],
        [
            greedy_pdr,
            qlearning_pdr,
        ],
    )

    plt.ylabel(
        "Packet Delivery Ratio (%)"
    )

    plt.title(
        "Packet Delivery Ratio Comparison"
    )

    plt.ylim(
        0,
        100,
    )

    save_graph(
        figure,
        "fig01_packet_delivery_ratio.png",
    )

    return qlearning_pdr


# ============================================================
# FIGURE 2
# PACKET LOSS
# ============================================================

def figure_2_packet_loss(
    greedy,
    qlearning,
):

    greedy_dropped = int(
        (
            ~greedy[
                "packet_delivered"
            ]
        ).sum()
    )

    qlearning_dropped = int(
        (
            ~qlearning[
                "packet_delivered"
            ]
        ).sum()
    )

    figure = plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        [
            "Greedy",
            "Q-learning",
        ],
        [
            greedy_dropped,
            qlearning_dropped,
        ],
    )

    plt.ylabel(
        "Packets Dropped"
    )

    plt.title(
        "Packet Loss Comparison"
    )

    save_graph(
        figure,
        "fig02_packet_loss.png",
    )

    return (
        greedy_dropped,
        qlearning_dropped,
    )


# ============================================================
# FIGURE 3
# AVERAGE HOP COUNT
# ============================================================

def figure_3_average_hops(
    greedy,
    qlearning,
):

    greedy_hops = float(
        greedy[
            "hop_count"
        ].mean()
    )

    qlearning_hops = float(
        qlearning[
            "hop_count"
        ].mean()
    )

    figure = plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        [
            "Greedy",
            "Q-learning",
        ],
        [
            greedy_hops,
            qlearning_hops,
        ],
    )

    plt.ylabel(
        "Average Hop Count"
    )

    plt.title(
        "Average Hop Count Comparison"
    )

    save_graph(
        figure,
        "fig03_average_hop_count.png",
    )

    return (
        greedy_hops,
        qlearning_hops,
    )


# ============================================================
# FIGURE 4
# HOP COUNT PER PACKET
# ============================================================

def figure_4_hop_count(
    greedy,
    qlearning,
):

    figure = plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        greedy[
            "packet_id"
        ],
        greedy[
            "hop_count"
        ],
        label="Greedy",
    )

    plt.plot(
        qlearning[
            "packet_id"
        ],
        qlearning[
            "hop_count"
        ],
        label="Q-learning",
    )

    plt.xlabel(
        "Packet ID"
    )

    plt.ylabel(
        "Hop Count"
    )

    plt.title(
        "Hop Count per Packet"
    )

    plt.legend()

    save_graph(
        figure,
        "fig04_hop_count_per_packet.png",
    )


# ============================================================
# FIGURE 5
# Q-LEARNING REWARD CONVERGENCE
# ============================================================

def figure_5_reward(
    training,
):

    reward_column = None

    for candidate in [
        "total_reward",
        "reward",
        "episode_reward",
        "cumulative_reward",
    ]:

        if candidate in training.columns:

            reward_column = candidate

            break

    if reward_column is None:

        raise KeyError(
            "\nCould not find the Q-learning reward "
            "column in q_learning_training.csv.\n\n"
            f"Actual columns:\n"
            f"{list(training.columns)}"
        )

    rewards = safe_numeric(
        training[
            reward_column
        ]
    )

    moving_rewards = (
        moving_average(
            rewards,
            MOVING_AVERAGE_WINDOW,
        )
    )

    figure = plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        training.index + 1,
        rewards,
        alpha=0.25,
        label="Episode Reward",
    )

    plt.plot(
        training.index + 1,
        moving_rewards,
        linewidth=2,
        label=(
            f"{MOVING_AVERAGE_WINDOW}-Episode "
            "Moving Average"
        ),
    )

    plt.xlabel(
        "Episode"
    )

    plt.ylabel(
        "Reward"
    )

    plt.title(
        "Q-Learning Reward Convergence"
    )

    plt.legend()

    save_graph(
        figure,
        "fig05_q_learning_reward_convergence.png",
    )


# ============================================================
# FIGURE 6
# TRAINING SUCCESS RATE
# ============================================================

def figure_6_training_success(
    training,
):

    success_column = None

    for candidate in [
        "delivered",
        "packet_delivered",
        "success",
        "successful",
        "episode_success",
        "route_success",
    ]:

        if candidate in training.columns:

            success_column = candidate

            break

    if success_column is None:

        raise KeyError(
            "\nCould not find the Q-learning training "
            "success column.\n\n"
            f"Expected one of:\n"
            f"  delivered\n"
            f"  packet_delivered\n"
            f"  success\n"
            f"  successful\n"
            f"  episode_success\n"
            f"  route_success\n\n"
            f"Actual columns:\n"
            f"{list(training.columns)}"
        )

    success = (
        training[
            success_column
        ]
        .apply(to_bool)
        .astype(float)
    )

    rolling_success = (
        success
        .rolling(
            window=MOVING_AVERAGE_WINDOW,
            min_periods=1,
        )
        .mean()
        * 100
    )

    figure = plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        training.index + 1,
        rolling_success,
        linewidth=2,
    )

    plt.xlabel(
        "Episode"
    )

    plt.ylabel(
        "Success Rate (%)"
    )

    plt.title(
        "Q-Learning Training Success Rate"
    )

    plt.ylim(
        0,
        100,
    )

    save_graph(
        figure,
        "fig06_q_learning_training_success.png",
    )


# ============================================================
# FIGURE 7
# EPSILON DECAY
# ============================================================

def figure_7_epsilon(
    training,
):

    epsilon_column = None

    for candidate in [
        "epsilon",
        "exploration_rate",
        "epsilon_value",
    ]:

        if candidate in training.columns:

            epsilon_column = candidate

            break

    if epsilon_column is None:

        print(
            "\nWARNING:"
        )

        print(
            "Epsilon column was not found."
        )

        print(
            "Figure 7 will be skipped."
        )

        print(
            f"Training CSV columns: "
            f"{list(training.columns)}"
        )

        return

    epsilon = safe_numeric(
        training[
            epsilon_column
        ]
    )

    figure = plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        training.index + 1,
        epsilon,
    )

    plt.xlabel(
        "Episode"
    )

    plt.ylabel(
        "Epsilon"
    )

    plt.title(
        "Q-Learning Epsilon Decay"
    )

    save_graph(
        figure,
        "fig07_epsilon_decay.png",
    )


# ============================================================
# FIGURE 8
# ATTACK DETECTION PERFORMANCE
# ============================================================

def figure_8_detection(
    metrics,
):

    figure = plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        metrics[
            "attack"
        ],
        metrics[
            "attack_detected"
        ],
    )

    plt.ylabel(
        "Detection Rate (%)"
    )

    plt.xlabel(
        "Attack Type"
    )

    plt.title(
        "Attack Detection Performance"
    )

    plt.ylim(
        0,
        100,
    )

    plt.xticks(
        rotation=20,
        ha="right",
    )

    save_graph(
        figure,
        "fig08_attack_detection_performance.png",
    )


# ============================================================
# FIGURE 9
# PACKET DELIVERY UNDER ATTACKS
# ============================================================

def figure_9_attack_pdr(
    metrics,
    normal_pdr,
):

    labels = [
        "Normal",
        *metrics[
            "attack"
        ].tolist(),
    ]

    values = [
        normal_pdr,
        *metrics[
            "pdr"
        ].tolist(),
    ]

    figure = plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        labels,
        values,
    )

    plt.ylabel(
        "Packet Delivery Ratio (%)"
    )

    plt.title(
        "Packet Delivery Ratio Under Security Attacks"
    )

    plt.ylim(
        0,
        100,
    )

    plt.xticks(
        rotation=20,
        ha="right",
    )

    save_graph(
        figure,
        "fig09_packet_delivery_under_attacks.png",
    )


# ============================================================
# FIGURE 10
# PACKET DROP RATIO
# ============================================================

def figure_10_drop_ratio(
    metrics,
):

    figure = plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        metrics[
            "attack"
        ],
        metrics[
            "drop_ratio"
        ],
    )

    plt.ylabel(
        "Packet Drop Ratio (%)"
    )

    plt.xlabel(
        "Attack Type"
    )

    plt.title(
        "Packet Drop Ratio Under Security Attacks"
    )

    plt.xticks(
        rotation=20,
        ha="right",
    )

    save_graph(
        figure,
        "fig10_packet_drop_ratio_under_attacks.png",
    )


# ============================================================
# FIGURE 11
# TRUST DEGRADATION
# ============================================================

def figure_11_trust(
    attack_data,
):

    figure = plt.figure(
        figsize=(10, 6)
    )

    plotted = False

    for attack_name, df in attack_data.items():

        if (
            "trust_after"
            not in df.columns
        ):
            continue

        trust = (
            safe_numeric(
                df[
                    "trust_after"
                ]
            )
            .dropna()
            .reset_index(
                drop=True
            )
        )

        if trust.empty:
            continue

        plt.plot(
            range(
                1,
                len(trust) + 1,
            ),
            trust,
            marker="o",
            label=attack_name,
        )

        plotted = True

    if not plotted:

        plt.close(
            figure
        )

        print(
            "\nWARNING:"
        )

        print(
            "No trust_after data found."
        )

        print(
            "Figure 11 skipped."
        )

        return

    plt.axhline(
        y=0.7,
        linestyle="--",
        label="Blacklist Threshold",
    )

    plt.xlabel(
        "Observed Trust Event"
    )

    plt.ylabel(
        "Trust Value"
    )

    plt.title(
        "Trust Degradation Under Security Attacks"
    )

    plt.ylim(
        0,
        1.05,
    )

    plt.legend()

    save_graph(
        figure,
        "fig11_trust_degradation.png",
    )


# ============================================================
# FIGURE 12
# BLACKLIST ACTIVATION
# ============================================================

def figure_12_blacklist(
    metrics,
):

    figure = plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        metrics[
            "attack"
        ],
        metrics[
            "blacklist_events"
        ],
    )

    plt.ylabel(
        "Blacklist Events"
    )

    plt.xlabel(
        "Attack Type"
    )

    plt.title(
        "Blacklist Activation Under Security Attacks"
    )

    plt.xticks(
        rotation=20,
        ha="right",
    )

    save_graph(
        figure,
        "fig12_blacklist_activation.png",
    )


# ============================================================
# FIGURE 13
# ATTACK IMPACT COMPARISON
# ============================================================

def figure_13_attack_impact(
    metrics,
):

    figure = plt.figure(
        figsize=(11, 6)
    )

    x = np.arange(
        len(metrics)
    )

    width = 0.25

    plt.bar(
        x - width,
        metrics[
            "dropped_packets"
        ],
        width,
        label="Dropped Packets",
    )

    plt.bar(
        x,
        metrics[
            "route_failures"
        ],
        width,
        label="Route Failures",
    )

    plt.bar(
        x + width,
        metrics[
            "blacklist_events"
        ],
        width,
        label="Blacklist Events",
    )

    plt.xticks(
        x,
        metrics[
            "attack"
        ],
        rotation=20,
        ha="right",
    )

    plt.ylabel(
        "Count"
    )

    plt.title(
        "Security Attack Impact Comparison"
    )

    plt.legend()

    save_graph(
        figure,
        "fig13_attack_impact_comparison.png",
    )


# ============================================================
# FIGURE 14
# ROUTE FAILURES
# ============================================================

def figure_14_route_failures(
    metrics,
):

    figure = plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        metrics[
            "attack"
        ],
        metrics[
            "route_failures"
        ],
    )

    plt.ylabel(
        "Route Failures"
    )

    plt.xlabel(
        "Attack Type"
    )

    plt.title(
        "Route Failures Under Security Attacks"
    )

    plt.xticks(
        rotation=20,
        ha="right",
    )

    save_graph(
        figure,
        "fig14_route_failures.png",
    )


# ============================================================
# FIGURE 15
# SECURITY VS ROUTING COST
# ============================================================

def figure_15_security_routing_cost(
    qlearning,
    metrics,
):

    normal_hops = float(
        qlearning[
            "hop_count"
        ].mean()
    )

    attack_labels = [
        "Normal",
        *metrics[
            "attack"
        ].tolist(),
    ]

    attack_hops = (
        metrics[
            "average_hops"
        ]
        .fillna(
            normal_hops
        )
        .tolist()
    )

    hop_values = [
        normal_hops,
        *attack_hops,
    ]

    figure = plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        attack_labels,
        hop_values,
    )

    plt.ylabel(
        "Average Hop Count"
    )

    plt.title(
        "Security Conditions vs Routing Cost"
    )

    plt.xticks(
        rotation=20,
        ha="right",
    )

    save_graph(
        figure,
        "fig15_security_vs_routing_cost.png",
    )


# ============================================================
# FIGURE 16
# OVERALL SECURITY METRICS
# ============================================================

def figure_16_security_summary(
    metrics,
):

    # --------------------------------------------------------
    # These remain separate indicators.
    #
    # We intentionally do NOT calculate an arbitrary overall
    # "security score".
    # --------------------------------------------------------

    detection = (
        metrics[
            "attack_detected"
        ]
    )

    pdr = (
        metrics[
            "pdr"
        ]
    )

    trust_retained = (
        metrics[
            "minimum_trust"
        ]
        * 100
    )

    blacklist_activation = (
        metrics[
            "blacklist_events"
        ]
        .apply(
            lambda value:
            100.0
            if value > 0
            else 0.0
        )
    )

    x = np.arange(
        len(metrics)
    )

    width = 0.20

    figure = plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        x - 1.5 * width,
        detection,
        width,
        label="Attack Detection",
    )

    plt.bar(
        x - 0.5 * width,
        pdr,
        width,
        label="Packet Delivery Ratio",
    )

    plt.bar(
        x + 0.5 * width,
        trust_retained,
        width,
        label="Minimum Trust Retained",
    )

    plt.bar(
        x + 1.5 * width,
        blacklist_activation,
        width,
        label="Blacklist Activation",
    )

    plt.xticks(
        x,
        metrics[
            "attack"
        ],
        rotation=20,
        ha="right",
    )

    plt.ylabel(
        "Percentage (%)"
    )

    plt.title(
        "Overall Security Evaluation"
    )

    plt.ylim(
        0,
        100,
    )

    plt.legend()

    save_graph(
        figure,
        "fig16_overall_security_metrics.png",
    )


# ============================================================
# SECURITY SUMMARY
# ============================================================

def print_security_summary(
    metrics,
):

    print(
        "\n"
        + "=" * 72
    )

    print(
        "              SECURITY COMPARISON SUMMARY"
    )

    print(
        "=" * 72
    )

    for _, row in metrics.iterrows():

        minimum_trust = (
            row[
                "minimum_trust"
            ]
        )

        if pd.isna(
            minimum_trust
        ):

            trust_text = "N/A"

        else:

            trust_text = (
                f"{minimum_trust:.4f}"
            )

        print(
            f"\n{row['attack']}"
        )

        print(
            f"  Packets generated : "
            f"{int(row['total_packets'])}"
        )

        print(
            f"  Packets delivered : "
            f"{int(row['delivered_packets'])}"
        )

        print(
            f"  Packets dropped   : "
            f"{int(row['dropped_packets'])}"
        )

        print(
            f"  Route failures    : "
            f"{int(row['route_failures'])}"
        )

        print(
            f"  Packet delivery   : "
            f"{row['pdr']:.2f}%"
        )

        print(
            f"  Packet drop ratio : "
            f"{row['drop_ratio']:.2f}%"
        )

        print(
            f"  Minimum trust     : "
            f"{trust_text}"
        )

        print(
            f"  Blacklist events  : "
            f"{int(row['blacklist_events'])}"
        )

        print(
            f"  Attack detected   : "
            f"{'YES' if row['attack_detected'] > 0 else 'NO'}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 72
    )

    print(
        "      RL-SRP RESEARCH GRAPH GENERATION"
    )

    print(
        "=" * 72
    )

    # ========================================================
    # LOAD ROUTING DATA
    # ========================================================

    (
        greedy,
        qlearning,
        training,
    ) = load_routing_data()

    # ========================================================
    # LOAD SECURITY DATA
    # ========================================================

    attack_data = (
        load_attack_results()
    )

    security_metrics = (
        calculate_attack_metrics(
            attack_data
        )
    )

    # ========================================================
    # ROUTING FIGURES
    # ========================================================

    qlearning_pdr = figure_1_pdr(
        greedy,
        qlearning,
    )

    (
        greedy_dropped,
        qlearning_dropped,
    ) = figure_2_packet_loss(
        greedy,
        qlearning,
    )

    (
        greedy_hops,
        qlearning_hops,
    ) = figure_3_average_hops(
        greedy,
        qlearning,
    )

    figure_4_hop_count(
        greedy,
        qlearning,
    )

    figure_5_reward(
        training
    )

    figure_6_training_success(
        training
    )

    figure_7_epsilon(
        training
    )

    # ========================================================
    # SECURITY FIGURES
    # ========================================================

    figure_8_detection(
        security_metrics
    )

    figure_9_attack_pdr(
        security_metrics,
        qlearning_pdr,
    )

    figure_10_drop_ratio(
        security_metrics
    )

    figure_11_trust(
        attack_data
    )

    figure_12_blacklist(
        security_metrics
    )

    figure_13_attack_impact(
        security_metrics
    )

    figure_14_route_failures(
        security_metrics
    )

    figure_15_security_routing_cost(
        qlearning,
        security_metrics,
    )

    figure_16_security_summary(
        security_metrics
    )

    # ========================================================
    # ROUTING SUMMARY
    # ========================================================

    greedy_pdr = (
        greedy[
            "packet_delivered"
        ]
        .mean()
        * 100
    )

    print(
        f"\nGreedy Packet Delivery Ratio : "
        f"{greedy_pdr:.2f}%"
    )

    print(
        f"Q-Learning Packet Delivery Ratio : "
        f"{qlearning_pdr:.2f}%"
    )

    print(
        f"\nGreedy Average Hop Count : "
        f"{greedy_hops:.2f}"
    )

    print(
        f"Q-Learning Average Hop Count : "
        f"{qlearning_hops:.2f}"
    )

    print(
        f"\nGreedy Packets Dropped : "
        f"{greedy_dropped}"
    )

    print(
        f"Q-Learning Packets Dropped : "
        f"{qlearning_dropped}"
    )

    print(
        f"\nTraining Episodes : "
        f"{len(training)}"
    )

    # ========================================================
    # SECURITY SUMMARY
    # ========================================================

    print_security_summary(
        security_metrics
    )

    # ========================================================
    # FINAL GRAPH LIST
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "              GRAPH GENERATION COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        "\nRouting figures:"
    )

    print(
        "  Fig1  : Packet Delivery Ratio Comparison"
    )

    print(
        "  Fig2  : Packet Loss Comparison"
    )

    print(
        "  Fig3  : Average Hop Count Comparison"
    )

    print(
        "  Fig4  : Hop Count Per Packet"
    )

    print(
        "  Fig5  : Q-Learning Reward Convergence"
    )

    print(
        "  Fig6  : Q-Learning Training Success Rate"
    )

    print(
        "  Fig7  : Epsilon Decay"
    )

    print(
        "\nSecurity figures:"
    )

    print(
        "  Fig8  : Attack Detection Performance"
    )

    print(
        "  Fig9  : Packet Delivery Ratio Under Attacks"
    )

    print(
        "  Fig10 : Packet Drop Ratio Under Attacks"
    )

    print(
        "  Fig11 : Trust Degradation Under Attacks"
    )

    print(
        "  Fig12 : Blacklist Activation"
    )

    print(
        "  Fig13 : Attack Impact Comparison"
    )

    print(
        "  Fig14 : Route Failures Under Attacks"
    )

    print(
        "  Fig15 : Security vs Routing Cost"
    )

    print(
        "  Fig16 : Overall Security Metrics"
    )

    print(
        "\nFigures saved to:"
    )

    print(
        f"  {FIGURE_DIR.resolve()}"
    )

    print(
        "\n"
        + "=" * 72
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()