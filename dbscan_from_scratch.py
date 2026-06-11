import numpy as np
import matplotlib.pyplot as plt

# داده‌های اولیه
np.random.seed(42)
n = 500
x_o = np.array([.3, -1, -1.5])
y_o = np.array([-.6, -1.5, 2])
x_b = np.random.normal(-.25, .1, n // 5)
y_b = np.random.normal(0, .1, n // 5)
theta = np.random.uniform(0, 10, n)
r = .5 + .15 * theta
x_s = r * np.cos(theta)
y_s = r * np.sin(theta) + np.random.normal(0, .1, n)
x_1 = np.hstack([x_o, x_b, x_s])
x_2 = np.hstack([y_o, y_b, y_s])

# Dataset X
X = np.vstack([x_1, x_2]).T


def find_neighbors(X, eps, point):
    distances = np.linalg.norm(X - X[point], axis=1)
    return np.where(distances <= eps)[0]

def expand_cluster(X, point, neighbors, eps, min_samples, visited, clusters, cluster_index):
    clusters[cluster_index].append(point)
    for neighbor in neighbors:
        if neighbor not in visited:
            visited.add(neighbor)
            new_neighbors = find_neighbors(X, eps, neighbor)
            if len(new_neighbors) >= min_samples:
                expand_cluster(X, neighbor, new_neighbors, eps, min_samples, visited, clusters, cluster_index)
        if neighbor not in clusters[cluster_index]:
            clusters[cluster_index].append(neighbor)

def DBSCAN(X, eps, min_samples):
    visited = set()
    clusters = []
    noise = []

    for point in range(len(X)):
        if point in visited:
            continue
        visited.add(point)
        neighbors = find_neighbors(X, eps, point)
        if len(neighbors) < min_samples:
            noise.append(point)
        else:
            clusters.append([])
            expand_cluster(X, point, neighbors, eps, min_samples, visited, clusters, len(clusters) - 1)
    
    return clusters, noise

eps = 0.2
min_samples = 5
clusters, noise = DBSCAN(X, eps, min_samples)

plt.figure(figsize=(8, 8))
colors = plt.cm.tab10(np.arange(len(clusters)))
for i, cluster in enumerate(clusters):
    cluster_points = X[np.array(cluster)]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}", color=colors[i % len(colors)])

noise_points = X[np.array(noise)]
plt.scatter(noise_points[:, 0], noise_points[:, 1], color='k', label='Noise')
plt.legend()
plt.grid(True)
plt.show()
