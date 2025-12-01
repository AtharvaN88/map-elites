from ptme_core.ptme_core import PTME
from domains.bipedal_walker import BipedalDomain
import numpy as np
import pickle

def fitness_biped(x, theta):
    env = BipedalDomain(render_mode=None)
    state = env.reset(theta)

    total_reward = 0.0
    steps = 600

    for _ in range(steps):
        _, r, done, trunc = env.step_cpg(x)
        total_reward += r
        if done or trunc:
            break

    # Normalize reward to [0,1]
    f = np.tanh(total_reward / 100)
    return float(max(0.0, f))

def main():

    ptme = PTME(
        d_theta=2,
        d_x=13,
        fitness_fn=fitness_biped,
        n_cells=100,
        budget=1000,
        rng=999
    )

    archive, evals = ptme.run()

    with open("data/biped/archive.pkl", "wb") as f:
        pickle.dump(archive, f)

    with open("data/biped/evals.pkl", "wb") as f:
        pickle.dump(evals, f)

    print("Finished tiny biped run!")
    print("Archive size:", len(archive))
    print("Evaluations:", len(evals))

if __name__ == "__main__":
    main()
