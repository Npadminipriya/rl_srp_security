# RL-SRP: Reinforcement Learning Based Secure Routing Protocol

RL-SRP (Reinforcement Learning Secure Routing Protocol) is a Python-based simulation framework for studying secure routing in simulated wireless multi-hop networks.

The framework combines **Q-Learning based routing** with **dynamic trust management** to learn forwarding paths while detecting and isolating malicious neighbour relationships.

The project evaluates routing performance and security behaviour under multiple attack scenarios, including **Blackhole, Selective Forwarding, Replay, and Packet Modification attacks**.

---

## Overview

The main objective of RL-SRP is to investigate whether reinforcement-learning-based route selection combined with trust-based security mechanisms can improve packet delivery and detect malicious forwarding behaviour in a simulated multi-hop network.

The system consists of:

* Network topology generation
* Energy-aware network modelling
* Dynamic neighbour trust management
* Q-Learning based route selection
* Malicious behaviour simulation
* Trust degradation
* Dynamic blacklisting
* Routing performance evaluation
* Security evaluation
* CSV-based experimental results
* Automated research graph generation

---

## Key Features

* Q-Learning based routing
* Dynamic trust management
* Directional neighbour trust evaluation
* Automatic malicious behaviour detection
* Dynamic blacklisting
* Blackhole attack simulation
* Selective forwarding attack simulation
* Replay attack simulation
* Packet modification attack simulation
* Greedy vs Q-Learning routing comparison
* Packet delivery and packet loss analysis
* Trust degradation analysis
* Blacklist activation analysis
* Q-Learning convergence analysis
* Automated research figure generation
* CSV result export

---

## System Architecture

```text
+-------------------------+
|      Source Node        |
+------------+------------+
             |
             v
+-------------------------+
|   Network State         |
|   Representation        |
+------------+------------+
             |
             v
+-------------------------+
|      Q-Learning         |
|    Route Selection      |
+------------+------------+
             |
             v
+-------------------------+
|   Trust Evaluation      |
|  Neighbour Assessment   |
+------------+------------+
             |
             v
+-------------------------+
|   Blacklist Checking    |
+------------+------------+
             |
             v
+-------------------------+
|    Packet Forwarding    |
+------------+------------+
             |
             v
+-------------------------+
|      Sink Node          |
+-------------------------+
```

---

## Network Configuration

The current experimental configuration uses:

| Parameter                        | Value |
| -------------------------------- | ----: |
| Sensor nodes                     |    50 |
| Sink node                        |     1 |
| Total network nodes              |    51 |
| Training episodes                | 5,000 |
| Final epsilon                    |  0.01 |
| Security test packets            |   250 |
| Packets per sensor               |     5 |
| Malicious node                   |    11 |
| Initial trust                    |   1.0 |
| Malicious threshold              |   0.7 |
| Selective forwarding probability |   50% |
| Replay probability               |   30% |
| Modification probability         |   30% |

The network is evaluated as a simulated wireless multi-hop environment. The project does not represent a deployment on physical wireless hardware.

---

## Project Structure

```text
rl_srp_security/
│
├── config.py
├── config.json
├── topology.py
├── trust_manager.py
├── q_learning.py
│
├── attack_models.py
├── selective_forwarding_attack.py
├── replay_attack.py
├── modification_attack.py
│
├── run_phase2.py
├── run_phase3.py
├── run_phase4.py
├── run_phase5.py
├── run_phase6.py
├── run_phase7.py
├── run_phase8.py
├── run_phase9.py
├── run_phase10.py
├── run_phase11.py
│
├── graph_gen.py
│
├── outputs/
│   ├── packet_forwarding_results.csv
│   ├── rl_packet_routing_results.csv
│   ├── q_learning_training.csv
│   ├── blackhole_attack_results.csv
│   ├── selective_forwarding_results.csv
│   ├── replay_attack_results.csv
│   ├── packet_modification_results.csv
│   │
│   └── research_figures/
│       ├── figure_01_pdr_comparison.png
│       ├── figure_02_packet_loss.png
│       ├── figure_03_average_hop_count.png
│       ├── figure_04_hop_count_per_packet.png
│       ├── figure_05_q_learning_reward.png
│       ├── figure_06_training_success_rate.png
│       ├── figure_07_epsilon_decay.png
│       ├── figure_08_attack_detection.png
│       ├── figure_09_attack_pdr.png
│       ├── figure_10_attack_drop_ratio.png
│       ├── figure_11_trust_degradation.png
│       ├── figure_12_blacklist_activation.png
│       ├── figure_13_attack_impact.png
│       ├── figure_14_route_failures.png
│       ├── figure_15_security_routing_cost.png
│       └── figure_16_security_summary.png
│
└── README.md
```

---

# Routing Mechanism

RL-SRP uses Q-Learning to learn forwarding decisions.

## State

The current network state is represented using routing and neighbour information available to the learning agent.

## Action

An action corresponds to selecting a neighbouring node as the next hop.

## Reward

The routing process uses rewards associated with forwarding behaviour, successful delivery, failed transmissions, and security-related behaviour.

The Q-Learning update is:

```text
Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

Where:

* `s` = current state
* `a` = selected action
* `r` = reward
* `α` = learning rate
* `γ` = discount factor

The current training configuration uses **5,000 episodes**.

---

# Trust Management

RL-SRP maintains directional trust relationships between neighbouring nodes.

```text
Trust(A, B)
```

represents the trust that node `A` has in neighbouring node `B`.

## Trust Update

Successful forwarding increases the observed trust value.

Failed or malicious forwarding decreases the observed trust value.

The current implementation uses:

```text
Successful forwarding:
    Trust += 0.01

Failed forwarding:
    Trust -= 0.10
```

Trust values are bounded between `0.0` and `1.0`.

## Blacklisting

A neighbour relationship is blacklisted when:

```text
Trust <= 0.7
```

Once blacklisted, the relationship is excluded from subsequent routing decisions.

---

# Routing Performance Evaluation

RL-SRP is evaluated against a Greedy routing approach using packet delivery, packet loss, and route length.

## Current Results

| Metric                | Greedy | Q-Learning |
| --------------------- | -----: | ---------: |
| Packet Delivery Ratio | 88.00% |    100.00% |
| Packets Dropped       |      6 |          0 |
| Average Hop Count     |   2.40 |       4.14 |
| Training Episodes     |      — |      5,000 |

In the current simulation, Q-Learning achieved a higher packet delivery ratio and fewer observed packet drops, while using a higher average hop count.

This indicates a trade-off between delivery reliability and route length in the evaluated network.

---

# Security Attack Models

The security evaluation uses four attack scenarios.

## Phase 8: Blackhole Attack

A malicious node attempts to drop every packet that reaches it.

```text
Source
   |
   v
Malicious Node
   |
   X
 DROP
```

The trust manager observes failed forwarding behaviour and decreases the trust value associated with the malicious neighbour.

### Validation

* Blackhole behaviour observed
* Trust degradation observed
* Blacklisting triggered

---

## Phase 9: Selective Forwarding Attack

A malicious node selectively drops packets according to a configured probability.

Current configuration:

```text
Drop probability = 50%
```

Conceptually:

```text
Packet 1 -> Forward
Packet 2 -> Drop
Packet 3 -> Forward
Packet 4 -> Drop
```

### Validation

* Selective forwarding behaviour observed
* Trust degradation observed
* Blacklisting triggered

---

## Phase 10: Replay Attack

The replay attack models malicious reuse of previously observed packet identifiers.

Current configuration:

```text
Replay probability = 30%
```

When a replay event is triggered, the current implementation records the malicious event and applies a trust penalty.

### Validation

* Replay event observed
* Trust degradation observed
* Blacklisting triggered

---

## Phase 11: Packet Modification Attack

The packet modification experiment models malicious packet modification behaviour as a detected forwarding failure.

Current configuration:

```text
Modification probability = 30%
```

The current implementation records a modification event and applies a trust penalty.

> **Implementation note:** The current experiment does not perform byte-level payload mutation. Therefore, the results should be interpreted as a simulated packet-modification detection event rather than a demonstration of actual packet-content alteration.

### Validation

* Modification behaviour observed
* Trust degradation observed
* Blacklisting triggered

---

# Security Evaluation

Each attack experiment uses:

```text
50 sensor nodes
5 packets per sensor
---------------------
250 test packets
```

The current results are:

| Attack               | Packets | Delivered | Dropped |    PDR | Drop Ratio | Minimum Trust | Blacklist |
| -------------------- | ------: | --------: | ------: | -----: | ---------: | ------------: | --------: |
| Blackhole            |     250 |       247 |       3 | 98.80% |      1.20% |        0.7000 |       Yes |
| Selective Forwarding |     250 |       247 |       3 | 98.80% |      1.20% |        0.7000 |       Yes |
| Replay               |     250 |       247 |       3 | 98.80% |      1.20% |        0.7000 |       Yes |
| Modification         |     250 |       247 |       3 | 98.80% |      1.20% |        0.7000 |       Yes |

The identical aggregate results across the four current attack experiments are a property of the current simulation configuration and should not be interpreted as evidence that the four attacks have identical real-world impact.

---

# Performance Metrics

## Packet Delivery Ratio

```text
PDR = (Packets Delivered / Packets Sent) × 100
```

## Packet Drop Ratio

```text
Drop Ratio = (Packets Dropped / Packets Sent) × 100
```

## Average Hop Count

Average number of forwarding hops used to deliver packets to the destination.

## Trust Degradation

Measures the reduction in neighbour trust caused by observed failed or malicious forwarding behaviour.

## Blacklist Activation

Measures malicious neighbour relationships that reach the configured trust threshold and become blacklisted.

## Training Success

Measures successful routing outcomes observed during Q-Learning training.

---

# Research Graph Generation

The project includes an automated graph-generation script:

```text
python graph_gen.py
```

The script reads the routing, training, and attack CSV files and generates **16 research figures**.

### Routing Figures

1. Packet Delivery Ratio Comparison
2. Packet Loss Comparison
3. Average Hop Count Comparison
4. Hop Count Per Packet
5. Q-Learning Reward Convergence
6. Q-Learning Training Success Rate
7. Epsilon Decay

### Security Figures

8. Attack Detection Performance
9. Packet Delivery Ratio Under Attacks
10. Packet Drop Ratio Under Attacks
11. Trust Degradation Under Attacks
12. Blacklist Activation
13. Attack Impact Comparison
14. Route Failures Under Attacks
15. Security vs Routing Cost
16. Overall Security Metrics

Generated figures are stored in:

```text
outputs/research_figures/
```

---

# Running the Project

Install the required Python packages:

```bash
pip install numpy pandas matplotlib
```

Run the individual experimental phases as required.

### Blackhole Attack

```bash
python run_phase8.py
```

### Selective Forwarding Attack

```bash
python run_phase9.py
```

### Replay Attack

```bash
python run_phase10.py
```

### Packet Modification Attack

```bash
python run_phase11.py
```

### Generate Research Figures

After the result CSV files have been generated:

```bash
python graph_gen.py
```

---

# Research Outputs

The experiment generates CSV files containing routing and security results.

Main outputs include:

```text
outputs/
├── packet_forwarding_results.csv
├── rl_packet_routing_results.csv
├── q_learning_training.csv
├── blackhole_attack_results.csv
├── selective_forwarding_results.csv
├── replay_attack_results.csv
└── packet_modification_results.csv
```

Research figures are stored separately under:

```text
outputs/research_figures/
```

---

# Current Experimental Status

The current implementation has completed the following experimental stages:

* Phase 2: Network topology
* Phase 3: Energy model
* Phase 4: Trust management
* Phase 5: State representation
* Phase 6: Reward function
* Phase 7: Q-Learning routing
* Phase 8: Blackhole attack
* Phase 9: Selective forwarding attack
* Phase 10: Replay attack
* Phase 11: Packet modification attack
* Research graph generation

Current Q-Learning training configuration:

```text
Training episodes : 5000
Final epsilon     : 0.010000
Q-table entries   : 340
Training success  : 98.60%
```

---

# Limitations

The current study is simulation-based and has several limitations:

* The evaluation uses a simulated network rather than physical wireless hardware.
* The current experiments use a single malicious node.
* The attack experiments use a fixed malicious node.
* The current packet-modification implementation does not perform byte-level payload mutation.
* The security experiments currently produce the same aggregate delivery/drop outcome across the four attack scenarios.
* Larger networks and multiple malicious nodes have not yet been evaluated.
* Statistical evaluation across multiple independent random seeds has not yet been included.

These limitations define possible directions for future experimentation.

---

# Future Enhancements

Possible future extensions include:

* Multiple malicious nodes
* Adaptive trust thresholds
* Multiple independent simulation runs
* Confidence intervals and statistical analysis
* Deep Reinforcement Learning (DQN)
* Sybil attack simulation
* Wormhole attack simulation
* More realistic network traffic models
* Packet integrity verification
* Detection-time analysis
* Attack severity analysis
* Larger network-scale evaluation
* Realistic mobility models
* NS-3 or hardware-based validation

---

# Technologies Used

* Python 3.x
* Reinforcement Learning
* Q-Learning
* Graph-Based Network Simulation
* Trust Management
* Wireless Multi-Hop Network Modelling
* CSV-Based Experimental Analysis
* NumPy
* Pandas
* Matplotlib

---

# Author

Developed as part of a secure routing and reinforcement learning research project focused on routing performance, malicious behaviour detection, trust management, and attack mitigation in simulated wireless multi-hop networks.
