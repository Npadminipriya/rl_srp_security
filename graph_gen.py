
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS
# ============================================================

OUTPUT_DIR = Path("outputs")
FIGURE_DIR = OUTPUT_DIR / "research_figures"

# Create figure directory automatically
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD SIMULATION RESULTS
# ============================================================

greedy = pd.read_csv(
    OUTPUT_DIR / "packet_forwarding_results.csv"
)

qlearning = pd.read_csv(
    OUTPUT_DIR / "rl_packet_routing_results.csv"
)

training = pd.read_csv(
    OUTPUT_DIR / "q_learning_training.csv"
)


# ============================================================
# 3. GRAPH SETTINGS
# ============================================================

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",

    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,

    "legend.fontsize": 10
})


def save_graph(fig, filename):
    """
    Save graph as a high-resolution PNG.
    """

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)


# ============================================================
# FIGURE 1
# PACKET DELIVERY RATIO
# ============================================================

greedy_pdr = greedy["delivered"].mean() * 100
qlearning_pdr = qlearning["delivered"].mean() * 100


fig, ax = plt.subplots(figsize=(6.5, 4.5))

ax.bar(
    ["Greedy", "Q-Learning"],
    [greedy_pdr, qlearning_pdr]
)

ax.set_xlabel("Routing Algorithm")
ax.set_ylabel("Packet Delivery Ratio (%)")

ax.set_title(
    "Packet Delivery Ratio Comparison"
)

ax.set_ylim(0, 105)

ax.grid(
    axis="y",
    alpha=0.25
)

save_graph(
    fig,
    "Fig1_packet_delivery_ratio.png"
)


# ============================================================
# FIGURE 2
# PACKET LOSS
# ============================================================

greedy_loss = greedy["dropped"].sum()

qlearning_loss = (
    ~qlearning["delivered"]
).sum()


fig, ax = plt.subplots(figsize=(6.5, 4.5))

ax.bar(
    ["Greedy", "Q-Learning"],
    [greedy_loss, qlearning_loss]
)

ax.set_xlabel("Routing Algorithm")
ax.set_ylabel("Packets Dropped")

ax.set_title(
    "Packet Loss Comparison"
)

ax.grid(
    axis="y",
    alpha=0.25
)

save_graph(
    fig,
    "Fig2_packet_loss.png"
)


# ============================================================
# FIGURE 3
# AVERAGE HOP COUNT
# ============================================================

greedy_hops = greedy.loc[
    greedy["delivered"],
    "hop_count"
].mean()

qlearning_hops = qlearning.loc[
    qlearning["delivered"],
    "hop_count"
].mean()


fig, ax = plt.subplots(figsize=(6.5, 4.5))

ax.bar(
    ["Greedy", "Q-Learning"],
    [greedy_hops, qlearning_hops]
)

ax.set_xlabel("Routing Algorithm")
ax.set_ylabel("Average Hop Count")

ax.set_title(
    "Average Hop Count for Delivered Packets"
)

ax.grid(
    axis="y",
    alpha=0.25
)

save_graph(
    fig,
    "Fig3_average_hop_count.png"
)


# ============================================================
# FIGURE 4
# HOP COUNT PER PACKET
# ============================================================

packet_id = np.arange(
    1,
    len(greedy) + 1
)


fig, ax = plt.subplots(figsize=(8, 4.8))

ax.plot(
    packet_id,
    greedy["hop_count"],
    marker="o",
    markersize=3,
    linewidth=1.2,
    label="Greedy"
)

ax.plot(
    packet_id,
    qlearning["hop_count"],
    marker="s",
    markersize=3,
    linewidth=1.2,
    label="Q-Learning"
)

ax.set_xlabel("Packet ID")
ax.set_ylabel("Hop Count")

ax.set_title(
    "Hop Count per Packet"
)

ax.legend()

ax.grid(
    alpha=0.25
)

save_graph(
    fig,
    "Fig4_hop_count_per_packet.png"
)


# ============================================================
# FIGURE 5
# Q-LEARNING REWARD CONVERGENCE
# ============================================================

training["reward_ma100"] = (
    training["total_reward"]
    .rolling(
        100,
        min_periods=1
    )
    .mean()
)


fig, ax = plt.subplots(figsize=(8, 4.8))

# Raw reward
ax.plot(
    training["episode"],
    training["total_reward"],
    linewidth=0.6,
    alpha=0.35,
    label="Episode Reward"
)

# Moving average
ax.plot(
    training["episode"],
    training["reward_ma100"],
    linewidth=2.0,
    label="100-Episode Moving Average"
)

ax.set_xlabel("Training Episode")
ax.set_ylabel("Total Reward")

ax.set_title(
    "Q-Learning Reward Convergence"
)

ax.legend()

ax.grid(
    alpha=0.25
)

save_graph(
    fig,
    "Fig5_qlearning_reward_convergence.png"
)


# ============================================================
# FIGURE 6
# Q-LEARNING TRAINING SUCCESS RATE
# ============================================================

training["success"] = (
    training["delivered"]
    .astype(int)
)


training["success_rate_100"] = (
    training["success"]
    .rolling(
        100,
        min_periods=1
    )
    .mean()
    * 100
)


fig, ax = plt.subplots(figsize=(8, 4.8))

ax.plot(
    training["episode"],
    training["success_rate_100"],
    linewidth=2
)

ax.set_xlabel("Training Episode")
ax.set_ylabel(
    "Delivery Success Rate (%)"
)

ax.set_title(
    "Q-Learning Training Success Rate"
)

ax.set_ylim(0, 105)

ax.grid(
    alpha=0.25
)

save_graph(
    fig,
    "Fig6_qlearning_training_success_rate.png"
)


# ============================================================
# FIGURE 7
# EPSILON DECAY
# ============================================================

fig, ax = plt.subplots(figsize=(8, 4.8))

ax.plot(
    training["episode"],
    training["epsilon"],
    linewidth=2
)

ax.set_xlabel("Training Episode")
ax.set_ylabel("Epsilon")

ax.set_title(
    "Epsilon Decay During Q-Learning Training"
)

ax.grid(
    alpha=0.25
)

save_graph(
    fig,
    "Fig7_epsilon_decay.png"
)


# ============================================================
# 4. PRINT SUMMARY
# ============================================================

print("\n========================================")
print("      GRAPH GENERATION COMPLETE")
print("========================================")

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
    f"{greedy_loss}"
)

print(
    f"Q-Learning Packets Dropped : "
    f"{qlearning_loss}"
)

print(
    f"\nTraining Episodes : "
    f"{len(training)}"
)

print(
    "\nFigures saved to:"
)

print(
    FIGURE_DIR.resolve()
)

print("\n========================================")