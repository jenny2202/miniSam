import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Ensure the directory containing miniSam.py is on the path
sys.path.append('/mnt/data')
from miniSam import MiniSam

# 1. Generate synthetic 2D data
data, _ = make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=1.0, random_state=0)

# 2. Initialize and train SOM
som = MiniSam(x=20, y=20, input_len=2, sigma=10.0, learning_rate=0.5,
              neighborhood_function="gaussian", random_seed=42)
som.random_weights_init(data)
som.train_random(data, num_iteration=1000)

# 3. Compute U-Matrix and plot
u_matrix = som.distance_map()
plt.figure(figsize=(6,6))
plt.imshow(u_matrix, interpolation='nearest')
plt.title("SOM U-Matrix")
plt.colorbar(label="Average Distance to Neighbors")
plt.xlabel("Neuron X")
plt.ylabel("Neuron Y")
plt.tight_layout()
plt.show()
