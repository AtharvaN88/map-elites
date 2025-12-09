import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load archive
archive = pickle.load(open("data/biped/archive.pkl", "rb"))

# Extract task parameters and fitnesses
thetas = []
fitnesses = []

for cell in archive.values():
    thetas.append(cell.theta)
    fitnesses.append(cell.f)

thetas = np.array(thetas)
fitnesses = np.array(fitnesses)

plt.figure(figsize=(6,5))
sc = plt.scatter(thetas[:,0], thetas[:,1], c=fitnesses, cmap="viridis", s=60)

plt.colorbar(sc, label="Fitness")
plt.xlabel("theta[0]  (slope parameter)")
plt.ylabel("theta[1]  (friction parameter)")
plt.title("BipedWalker: Archive Fitness Map")

plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
