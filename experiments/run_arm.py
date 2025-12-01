from ptme_core.ptme_core import PTME
from domains.arm_10dof import fitness_arm
import numpy as np
import pickle

def main():
    # dimensions
    d_theta = 2
    d_x = 10

    ptme = PTME(
        d_theta=d_theta,
        d_x=d_x,
        fitness_fn=fitness_arm,
        n_cells=200,
        budget=5000,  # Can increase cells/budget but this is good enough for testing
        rng=42
    )

    archive, evaluations = ptme.run()

    # Save results
    with open("data/arm/arm_archive.pkl", "wb") as f:
        pickle.dump(archive, f)

    with open("data/arm/arm_evals.pkl", "wb") as f:
        pickle.dump(evaluations, f)

    print("Done! Archive size:", len(archive))
    print("Total evaluations:", len(evaluations))

if __name__ == "__main__":
    main()
