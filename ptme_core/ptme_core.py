import numpy as np
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans

class ArchiveCell:
    def __init__(self, theta_c, x, f, adj=None):
        self.theta = theta_c      # centroid task parameter
        self.x = x                # solution vector
        self.f = f                # fitness
        self.adj = adj or []      # adjacency list of indices


class PTME:
    """
    Minimal Parametric-Task MAP-Elites implementation (Algorithm 1)
    faithful to the paper's structure.
    """
    def __init__(self,
                 d_theta,
                 d_x,
                 fitness_fn,
                 n_cells=200,
                 budget=100_000,
                 S=[1, 5, 10, 50, 100, 500],
                 sigma_sbx=10.0,
                 sigma_reg=1.0,
                 rng=None):

        self.d_theta = d_theta
        self.d_x = d_x
        self.fitness_fn = fitness_fn

        self.n_cells = n_cells
        self.budget = budget
        self.S = S
        self.sigma_sbx = sigma_sbx
        self.sigma_reg = sigma_reg

        self.rng = np.random.default_rng(rng)

        self.archive = {}
        self.E = []  # all evaluations (theta, x, f)

        # tournament selection stats for UCB1
        self.selected = np.zeros(len(S), dtype=np.int32)
        self.successes = np.zeros(len(S), dtype=np.int32)


    # # # Initialization: CVT + KDTree # # #

    def initialize_archive(self):
        # Sample many points uniformly then cluster → centroids
        samples = self.rng.uniform(0, 1, size=(5000, self.d_theta))
        kmeans = KMeans(n_clusters=self.n_cells, n_init="auto")
        kmeans.fit(samples)
        centroids = kmeans.cluster_centers_

        self.centroids = centroids
        self.kdt = KDTree(centroids)

        # Initialize each cell with a random solution
        for i, theta_c in enumerate(centroids):
            x0 = self.rng.uniform(0, 1, size=self.d_x)
            f0 = self.fitness_fn(x0, theta_c)
            self.archive[i] = ArchiveCell(theta_c, x0, f0)

        # Simple adjacency: kNN = 10 neighbors
        # PT-ME Uses Delaunay triangulation; but this should be a simpler approximation
        n_neighbors = min(10, self.n_cells)
        _, idx = self.kdt.query(centroids, k=n_neighbors)
        for i in range(self.n_cells):
            self.archive[i].adj = list(idx[i][1:])  # exclude itself


    # # # Variation Operator 1: SBX crossover # # #

    def sbx(self, p1, p2):
        # Simulated Binary Crossover (simple version)
        u = self.rng.uniform(0, 1, size=self.d_x)
        beta = np.where(u <= 0.5,
                        (2*u)**(1/(self.sigma_sbx + 1)),
                        (1/(2*(1-u)))**(1/(self.sigma_sbx + 1)))
        child = 0.5*((1+beta)*p1 + (1-beta)*p2)
        child = np.clip(child, 0, 1)
        return child


    # # # Variation Operator 2: Local linear regression # # #

    def regression_operator(self, theta, cell_idx):
        neighbors = self.archive[cell_idx].adj
        if len(neighbors) < 2:
            # fallback on random solution
            return self.rng.uniform(0, 1, size=self.d_x)

        Thetas = np.array([self.archive[j].theta for j in neighbors])
        Xs = np.array([self.archive[j].x for j in neighbors])

        # Least-squares: M = (ΘᵀΘ)^-1 Θᵀ X
        try:
            M, _, _, _ = np.linalg.lstsq(Thetas, Xs, rcond=None)
            x_pred = theta @ M
        except np.linalg.LinAlgError:
            x_pred = self.rng.uniform(0, 1, size=self.d_x)

        # Add noise based on variance of neighbors
        noise = self.sigma_reg * self.rng.normal(
            0,
            np.sqrt(np.var(Xs, axis=0) + 1e-8),
            size=self.d_x
        )

        x_new = np.clip(x_pred + noise, 0, 1)
        return x_new


    # # # UCB1 tournament-size selection # # #
    
    def choose_tournament_size(self):
        # Compute UCB1 scores
        total = np.sum(self.selected) + 1
        means = self.successes / (self.selected + 1e-8)
        bonuses = np.sqrt(2 * np.log(total) / (self.selected + 1e-8))
        return np.argmax(means + bonuses)


    # # # Main loop # # #
    def run(self):
        self.initialize_archive()

        # Pre-select random tournament size
        s_idx = self.rng.integers(len(self.S))

        for _ in range(self.budget - self.n_cells):

            # Decide which operator to use
            if self.rng.random() < 0.5:
                # SBX with tournament
                p1_idx, p2_idx = self.rng.integers(self.n_cells, size=2)
                p1 = self.archive[p1_idx]
                p2 = self.archive[p2_idx]

                s = self.S[s_idx]
                self.selected[s_idx] += 1

                # Sample candidate tasks
                task_samples = self.rng.uniform(0, 1, size=(s, self.d_theta))
                # pick the one closest to p1.theta
                dists = np.linalg.norm(task_samples - p1.theta, axis=1)
                theta = task_samples[np.argmin(dists)]

                # assign to nearest cell
                cell_idx = self.kdt.query([theta], k=1)[1][0][0]

                # SBX
                x = self.sbx(p1.x, p2.x)
                used_tournament = True

            else:
                # Linear regression operator
                theta = self.rng.uniform(0, 1, size=self.d_theta)
                cell_idx = self.kdt.query([theta], k=1)[1][0][0]
                x = self.regression_operator(theta, cell_idx)
                used_tournament = False

            # Evaluate
            f = self.fitness_fn(x, theta)
            self.E.append((theta, x, f))

            # Update archive
            if f >= self.archive[cell_idx].f:
                self.archive[cell_idx].theta = theta
                self.archive[cell_idx].x = x
                self.archive[cell_idx].f = f

                # reward bandit
                if used_tournament:
                    self.successes[s_idx] += 1

            # update UCB1 tournament size
            if used_tournament:
                s_idx = self.choose_tournament_size()

        return self.archive, self.E
