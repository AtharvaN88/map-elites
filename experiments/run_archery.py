from ptme_core.ptme_core import PTME
from domains.archery import fitness_archery
import pickle

def main():

    ptme = PTME(
        d_theta=2,
        d_x=2,
        fitness_fn=fitness_archery,
        n_cells=200,
        budget=5000,
        rng=123
    )

    archive, evals = ptme.run()

    with open("data/archery/archery_archive.pkl", "wb") as f:
        pickle.dump(archive, f)

    with open("data/archery/archery_evals.pkl", "wb") as f:
        pickle.dump(evals, f)

    print("Finished archery run:")
    print("  Archive size:", len(archive))
    print("  Evaluations :", len(evals))


if __name__ == "__main__":
    main()
