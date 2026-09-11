# RL-SRP: Reinforcement Learning Based Secure Routing Protocol

## Overview

RL-SRP (Reinforcement Learning Secure Routing Protocol) is a Python-based simulation framework that combines **Q-Learning** and **Trust Management** to provide secure routing in wireless multi-hop networks such as Wireless Sensor Networks (WSNs) and Mobile Ad Hoc Networks (MANETs).

The system learns optimal routing paths through reinforcement learning while detecting and isolating malicious nodes using a decentralized trust evaluation mechanism.

The framework also simulates multiple routing attacks and evaluates the effectiveness of trust-based mitigation techniques.

---

## Features

* Q-Learning based routing
* Dynamic trust management
* Decentralized neighbour trust evaluation
* Automatic malicious node detection
* Blacklisting mechanism
* Attack simulation framework
* Performance evaluation metrics
* CSV result export

---

## Architecture

```text
+-------------------+
|   Source Node     |
+-------------------+
          |
          v
+-------------------+
|   Q-Learning      |
| Route Selection   |
+-------------------+
          |
          v
+-------------------+
| Trust Evaluation  |
+-------------------+
          |
          v
+-------------------+
| Packet Forwarding |
+-------------------+
          |
          v
+-------------------+
| Sink/Base Station |
+-------------------+
```

---

## Project Structure

```text
rl_srp_security/
│
├── config.py
├── topology.py
├── trust_manager.py
├── q_learning.py
│
├── blackhole_attack.py
├── selective_forwarding_attack.py
├── replay_attack.py
├── modification_attack.py
│
├── run_phase8.py
├── run_phase9.py
├── run_phase10.py
├── run_phase11.py
│
├── outputs/
│   ├── blackhole_attack_results.csv
│   ├── selective_forwarding_results.csv
│   ├── replay_attack_results.csv
│   └── packet_modification_results.csv
│
└── README.md
```

---

## Routing Mechanism

The routing protocol uses **Q-Learning** to learn optimal forwarding paths.

### State

Current node in the network.

### Action

Selecting the next-hop neighbour.

### Reward

* Positive reward for successful forwarding.
* Positive reward for reaching the sink node.
* Negative reward for failed transmissions.
* Negative reward for malicious behaviour.

### Q-Learning Update

$$
Q(s,a)=Q(s,a)+\alpha[r+\gamma \max Q(s',a')-Q(s,a)]
$$

Where:

* \(s\) = current state
* \(a\) = action
* \(r\) = reward
* \(\alpha\) = learning rate
* \(\gamma\) = discount factor

---

## Trust Management

Each node maintains trust values for its neighbours.

```text
Trust(A, B)
```

represents the trust that node A has in node B.

### Trust Update

Successful forwarding:

```text
Trust increases
```

Failed or malicious forwarding:

```text
Trust decreases
```

### Blacklisting

A neighbour is blacklisted when:

```text
Trust <= Malicious Threshold
```

Blacklisted nodes are excluded from future routing decisions.

---

## Implemented Attacks

### Phase 8: Blackhole Attack

A malicious node drops every packet it receives.

```text
Source -> Malicious Node -> DROP
```

Validation:

* Attack detection
* Trust degradation
* Blacklisting

---

### Phase 9: Selective Forwarding Attack

A malicious node drops packets with a predefined probability.

```text
Forward
Forward
Drop
Forward
Drop
```

Validation:

* Attack detection
* Trust degradation
* Blacklisting

---

### Phase 10: Replay Attack

A malicious node retransmits previously observed packets.

```text
Packet 15
      |
Store Packet
      |
Replay Packet 15
```

Validation:

* Replay detection
* Trust degradation
* Blacklisting

---

### Phase 11: Packet Modification Attack

A malicious node alters packet contents before forwarding.

```text
Receive Packet
      |
Modify Data
      |
Forward Modified Packet
```

Validation:

* Modification detection
* Trust degradation
* Blacklisting

---

## Performance Metrics

### Packet Delivery Ratio (PDR)

$$
PDR=
\frac{Packets\ Delivered}
{Packets\ Sent}
\times100
$$

---

### Packet Drop Ratio

$$
DropRatio=
\frac{Packets\ Dropped}
{Packets\ Sent}
\times100
$$

---

### Trust Degradation

Measures reduction in trust values caused by malicious behaviour.

---

### Blacklisting Events

Number of malicious neighbour relationships identified and isolated.

---

## Running the Project

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

---

## Sample Output

```text
RL-SRP PHASE 11: PACKET MODIFICATION ATTACK

Packets generated       : 50
Packets delivered       : 49
Packets dropped         : 1

Packet delivery ratio   : 98.00%
Packet drop ratio       : 2.00%

Blacklisted relationships: 1

Phase 11 validation

Packet modification observed : PASSED
Trust degradation            : PASSED
Blacklisting triggered       : PASSED
```

---

## Future Enhancements

* Adaptive trust thresholds
* Deep Reinforcement Learning (DQN)
* Multiple malicious nodes
* Sybil attack simulation
* Wormhole attack simulation
* Detection-time analysis
* Packet integrity metrics
* Realistic network traffic models
* Performance visualization and graph generation

---

## Technologies Used

* Python 3.x
* Reinforcement Learning (Q-Learning)
* Graph-Based Network Simulation
* Trust Management Systems
* CSV-Based Result Analysis

---

## Author

Developed as part of a secure routing and reinforcement learning research project focusing on attack detection and mitigation in wireless multi-hop networks.
