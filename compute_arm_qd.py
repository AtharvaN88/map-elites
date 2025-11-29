import pickle
import matplotlib.pyplot as plt
from metrics import multi_resolution_qd

D_THETA = 2  # 10-DoF Arm has 2 task parameters; joint angle and link length

def main():
    with open("arm_evals.pkl", "rb") as f:
        evals = pickle.load(f)

    resolutions, qd_scores, mr_qd = multi_resolution_qd(
        evals,
        d_theta=D_THETA,
        max_cells=len(evals), #10_000,      # raise this as budget grows; Paper says 'We cannot fill more cells than the budget'
        n_resolutions=30,
        rng=42
    )

    print("Multi-Resolution QD-Score (Arm):", mr_qd)

    # Plot QD-Score vs resolution (like Fig. 2c in the paper)
    plt.figure()
    plt.semilogx(resolutions, qd_scores, marker="o")
    plt.xlabel("Resolution (number of cells in archive)")
    plt.ylabel("QD-Score")
    plt.title("10-DoF Arm: QD-Score vs Resolution")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
