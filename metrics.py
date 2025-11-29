import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree


def build_archive_from_evals(evals, d_theta, n_cells, rng=None):
    """
    Rebuild an archive at a given resolution from a list of evaluations.

    evals   : list of (theta, x, f)
    d_theta : dimension of task parameters
    n_cells : number of CVT cells (archive resolution)

    Returns: list of cells, each dict has keys: theta_c, theta, x, f
    """
    rng = np.random.default_rng(rng)

    # Extract all task parameters
    thetas = np.array([t for (t, _, _) in evals])

    # CVT via k-means over theta space
    kmeans = KMeans(
        n_clusters=n_cells,
        n_init="auto",
        random_state=int(rng.integers(1_000_000_000))
    )
    kmeans.fit(thetas)
    centroids = kmeans.cluster_centers_

    # KDTree for fast cell lookup
    kdt = KDTree(centroids)

    # Initialize archive: f = 0, no solution yet
    archive = []
    for c in centroids:
        archive.append({
            "theta_c": c,      # centroid of this cell
            "theta": None,     # best task parameter seen in this cell
            "x": None,         # best solution
            "f": 0.0           # best fitness (assumes f >= 0)
        })

    # Go through all evaluations and keep the best per cell
    for theta, x, f in evals:
        # find nearest centroid
        cell_idx = kdt.query([theta], k=1)[1][0][0]
        if f >= archive[cell_idx]["f"]:
            archive[cell_idx]["theta"] = theta
            archive[cell_idx]["x"] = x
            archive[cell_idx]["f"] = float(f)

    return archive


def qd_score(archive):
    """
    QD-Score = sum of fitness over all filled cells.
    Empty cells contribute 0 by construction.

    archive: list of dicts returned by build_archive_from_evals
    """
    return sum(cell["f"] for cell in archive)


def multi_resolution_qd(evals, d_theta, max_cells=100_000, n_resolutions=50, rng=None):
    """
    Compute QD-Score across multiple resolutions and return:

    - resolutions: array of N_cell values
    - qd_scores : QD-Score at each resolution
    - mr_qd     : Multi-Resolution QD-Score (mean over resolutions)

    Resolutions are log-spaced from 1 to max_cells.
    """
    rng = np.random.default_rng(rng)

    # Log-spaced resolutions: 1 ... max_cells (integers, unique, sorted)
    resolutions = np.unique(
        np.logspace(0, np.log10(max_cells), num=n_resolutions, dtype=int)
    )

    qd_scores = []

    for n_cells in resolutions:
        archive = build_archive_from_evals(
            evals, d_theta=d_theta, n_cells=n_cells,
            rng=int(rng.integers(1_000_000_000))
        )
        qd_scores.append(qd_score(archive))

    qd_scores = np.array(qd_scores)
    mr_qd = float(np.mean(qd_scores))

    return resolutions, qd_scores, mr_qd
