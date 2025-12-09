# PT-ME: Parametric Task MAP-Elites (Re-Implementation)

This repository contains a research-oriented reimplementation of the **Parametric Task MAP-Elites (PT-ME)** algorithm, based on the paper:

**https://arxiv.org/abs/2402.01275**


The project recreates the core PT-ME algorithm from scratch in Python, reproduces results on the paper’s original benchmark domains (10-DoF Arm and Archery), and extends the study to an additional continuous-control domain (BipedalWalker-v3).

---

## Project Structure

```text
map-elites/
  ptme_core/
    ptme_core.py          # Core PT-ME algorithm (CVT archive, variation, regression)
    __init__.py

  domains/
    arm_10dof.py          # 10-DoF Arm kinematics + fitness
    archery.py            # Archery projectile simulation + fitness
    bipedal_walker.py     # BipedWalker-v3 wrapper + CPG controller
    __init__.py

  experiments/
    run_arm.py            # Run PT-ME on the Arm domain
    run_archery.py        # Run PT-ME on the Archery domain
    run_biped.py          # Run PT-ME on the Biped domain
    compute_arm_qd.py     # Compute QD / MR-QD for the Arm domain
    compute_archer_qd.py  # Compute QD / MR-QD for the Archery domain
    render_biped.py       # Renders a video of the biped; episode length and other parameters can be changed in this file
    __init__.py

  metrics/
    metrics.py            # QD-Score and Multi-Resolution QD-Score
    __init__.py

  plots/
    plot_arm.py           # Visualize the Arm archive in task space
    plot_archery.py       # Visualize the Archery archive
    plot_biped.py         # Visualize the Biped archive
    __init__.py

  data/
    arm/                  # Saved archives + eval logs for Arm
    archery/              # Saved archives + eval logs for Archery
    biped/                # Saved archives + eval logs for Biped

  videos/                 # (Generated) rollout videos, e.g. biped_demo.mp4

  requirements.txt
  README.md
```

---

## Environment Setup

Recommended conda setup:

```bash
conda create -n elites python=3.10
conda activate elites
pip install -r requirements.txt
```

Key dependencies include:

* `gymnasium` (environments)
* `box2d==2.3.10` (Box2D engine, WSL-friendly)
* `faiss-cpu` (fast CVT / k-means)
* `imageio`, `imageio-ffmpeg` (video export)
* `stable-baselines3`, `cma` (optional baselines)

---

## Running Experiments

All scripts are intended to be run from the project root using the module flag (`-m`) so imports resolve correctly.

### 10-DoF Arm

Run PT-ME on the Arm domain:

```bash
python -m experiments.run_arm
```

This will create:

```text
data/arm/archive.pkl
data/arm/evals.pkl
```

Plot the task-space archive:

```bash
python -m plots.plot_arm
```

Compute QD-Score and Multi-Resolution QD-Score:

```bash
python -m experiments.compute_arm_qd
```

---

### Archery

Run PT-ME on the Archery domain:

```bash
python -m experiments.run_archery
```

Plot the archive:

```bash
python -m plots.plot_archery
```

---

### BipedWalker-v3

Run PT-ME on the Biped domain (BipedWalker-v3 with a CPG controller):

```bash
python -m experiments.run_biped
```

Plot the task-space archive:

```bash
python -m plots.plot_biped
```

Render a short rollout video of the best controller:

```bash
python -m experiments.render_biped
```

This will create, for example:

```text
videos/biped_demo.mp4
```

---

## Domains Overview (Short)

* **10-DoF Arm (`arm_10dof.py`)**

  * Task parameters: 2D vector describing the target in workspace
  * Solution: 10 joint-angle parameters
  * Fitness: distance between end-effector and target (converted to a reward)

* **Archery (`archery.py`)**

  * Task parameters: target distance and wind strength
  * Solution: 2D aiming angles (yaw, pitch)
  * Fitness: negative squared miss distance (converted to a reward)

* **BipedWalker-v3 (`bipedal_walker.py`)**

  * Task parameters: slope and ground friction
  * Solution: 13D CPG-based controller (amplitudes, phases, biases, frequency)
  * Fitness: normalized cumulative reward from the Gymnasium environment

---

## Metrics

Implemented in `metrics/metrics.py`:

* **QD-Score**
  Sum of elite fitness values in a CVT archive at a given resolution.

* **Multi-Resolution QD-Score (MR-QD)**
  QD-Score computed over multiple archive resolutions (log-spaced) and averaged, following the evaluation protocol in the PT-ME paper.
