import pickle
import numpy as np
import matplotlib.pyplot as plt

archive = pickle.load(open("arm_archive.pkl", "rb"))

# Extract tasks and fitnesses
thetas = []
fits = []
for cell in archive.values():
    thetas.append(cell.theta)
    fits.append(cell.f)

thetas = np.array(thetas)
fits = np.array(fits)

# Heatmap-style scatter over the task space Θ
plt.scatter(thetas[:,0], thetas[:,1], c=fits, cmap='viridis')
plt.colorbar(label='Fitness')
plt.xlabel('θ[0] (max angle)')
plt.ylabel('θ[1] (link scale)')
plt.title("10-DoF Arm: Archive Fitness Distribution")
plt.show()
