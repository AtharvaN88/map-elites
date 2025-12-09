# PT-ME: Parametric Task MAP-Elites (Re-Implementation)

This repository contains a research-oriented reimplementation of the **Parametric Task MAP-Elites (PT-ME)** algorithm, based on the paper:

**https://arxiv.org/abs/2402.01275**


The project recreates the core PT-ME algorithm from scratch in Python, reproduces results on the paper’s original benchmark domains (10-DoF Arm and Archery), and extends the study to an additional continuous-control domain (BipedalWalker-v3).

---

## Project Structure

map-elites/
│
├── ptme_core/
│ ├── ptme_core.py # Core PT-ME algorithm (CVT archive, SBX + tournament, regression operator)
│ └── init.py
│
├── domains/
│ ├── arm_10dof.py # 10-DoF Arm forward-kinematics + fitness
│ ├── archery.py # Archery projectile simulation + reward
│ ├── bipedal_walker.py # Wrapper for BipedWalker-v3 + CPG controller
│ └── init.py
│
├── experiments/
│ ├── run_arm.py # Run PT-ME on Arm
│ ├── run_archery.py # Run PT-ME on Archery
│ ├── run_biped.py # Run PT-ME on Biped
│ ├── compute_arm_qd.py # Compute QD & MR-QD metrics using eval logs
│ └── init.py
│
├── metrics/
│ ├── metrics.py # QD-Score and Multi-Resolution QD implementation
│ └── init.py
│
├── plots/
│ ├── plot_arm.py # Visualize Arm archive in task space
│ ├── plot_archery.py # Visualize Archery archive
│ ├── plot_biped.py # Visualize Biped archive
│ └── init.py
│
├── data/
│ ├── arm/ # Saved archives + eval logs for Arm experiments
│ ├── archery/ # Archives + logs for Archery
│ └── biped/ # Archives + logs for Biped
│
├── videos/ # (Generated) BipedWalker rollout videos
│
├── requirements.txt # Environment setup
└── README.md
