# PT-ME: Parametric Task MAP-Elites (Re-Implementation)

This repository contains a research-oriented reimplementation of the **Parametric Task MAP-Elites (PT-ME)** algorithm, based on the paper:

**https://arxiv.org/abs/2402.01275**


The project recreates the core PT-ME algorithm from scratch in Python, reproduces results on the paper’s original benchmark domains (10-DoF Arm and Archery), and extends the study to an additional continuous-control domain (BipedalWalker-v3).

---

## Project Structure

map-elites/
│
├── ptme_core/
│ ├── ptme_core.py # Core PT-ME algorithm: CVT, SBX, regression operator, archive
│ └── init.py
│
├── domains/
│ ├── arm_10dof.py # 10-DoF Arm fitness function & kinematics
│ ├── archery.py # Ballistic archery domain
│ ├── bipedal_walker.py # BipedWalker-v3 wrapper + CPG controller
│ └── init.py
│
├── experiments/
│ ├── run_arm.py # Runs PT-ME on the Arm domain
│ ├── run_archery.py # Runs PT-ME on the Archery domain
│ ├── run_biped.py # Runs PT-ME on BipedWalker
│ ├── compute_arm_qd.py # QD + Multi-Resolution QD scoring (Arm)
│ └── init.py
│
├── metrics/
│ ├── metrics.py # QD-Score & MR-QD-Score implementation
│ └── init.py
│
├── plots/
│ ├── plot_arm.py # Task-space scatter plot for Arm archive
│ ├── plot_archery.py # Task-space scatter plot for Archery archive
│ ├── plot_biped.py # Task-space scatter plot for Biped archive
│ └── init.py
│
├── data/
│ ├── arm/ # Archive + eval logs for Arm runs
│ ├── archery/ # Archive + eval logs for Archery runs
│ └── biped/ # Archive + eval logs for Biped runs
│
└── requirements.txt # Python environment dependencies
