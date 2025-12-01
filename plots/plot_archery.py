import pickle
import numpy as np
import matplotlib.pyplot as plt

archive = pickle.load(open("archery_archive.pkl", "rb"))

T = np.array([c.theta for c in archive.values()])
F = np.array([c.f for c in archive.values()])

plt.scatter(T[:,0], T[:,1], c=F, cmap="viridis")
plt.xlabel("distance θ[0]")
plt.ylabel("wind θ[1]")
plt.colorbar(label="fitness")
plt.title("Archery: Fitness Map")
plt.show()