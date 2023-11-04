import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.decomposition import PCA

# Generate some sample data (You can replace this with your dataset)
n_samples = 300
n_features = 2
n_clusters = 4

data, true_labels = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)
data = StandardScaler().fit_transform(data)

# Define a K-means function to obtain cluster labels
def kmeans_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    return labels

# Calculate centroids for each cluster
def calculate_centroids(data, labels, n_clusters):
    centroids = np.zeros((n_clusters, n_features))
    for i in range(n_clusters):
        centroids[i] = np.mean(data[labels == i], axis=0)
    return centroids

# Calculate pairwise distances between cluster centroids
def calculate_centroid_distances(centroids):
    return pairwise_distances(centroids, centroids)

# Calculate Davies-Bouldin index
def davies_bouldin_index(data, labels, n_clusters, centroid_distances):
    db_indices = []

    for i in range(n_clusters):
        max_r = 0
        for j in range(n_clusters):
            if i != j:
                r = (calculate_si(data, labels, n_clusters, i) + calculate_si(data, labels, n_clusters, j)) / centroid_distances[i, j]
                max_r = max(max_r, r)
        db_indices.append(max_r)

    return np.mean(db_indices)

# Calculate S(i) for a cluster
def calculate_si(data, labels, n_clusters, cluster_idx):
    cluster_data = data[labels == cluster_idx]
    if len(cluster_data) == 0:
        return 0
    centroid = np.mean(cluster_data, axis=0)
    distances = np.linalg.norm(cluster_data - centroid, axis=1)
    return np.mean(distances)

# Plot the clustering results
def plot_clusters(data, labels, centroids):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data[:, 0], data[:, 1], hue=labels, palette="tab10")
    sns.scatterplot(centroids[:, 0], centroids[:, 1], color='black', s=100, label='Centroids')
    plt.legend(loc='best')
    plt.title("K-means Clustering Results")
    plt.show()

def main():
    # Load and preprocess your dataset here or use the generated data
    # df = pd.read_csv("your_dataset.csv")
    # Perform necessary preprocessing on your data

    # Use K-means clustering to obtain cluster labels
    labels = kmeans_clustering(data, n_clusters)

    # Calculate centroids for each cluster
    centroids = calculate_centroids(data, labels, n_clusters)

    # Calculate pairwise distances between cluster centroids
    centroid_distances = calculate_centroid_distances(centroids)

    # Calculate the Davies-Bouldin index
    db_index = davies_bouldin_index(data, labels, n_clusters, centroid_distances)
    print(f"The value of Davies-Bouldin index for a K-Means cluster of size {n_clusters} is: {db_index:.4f}")

    # Plot the clustering results
    plot_clusters(data, labels, centroids)

if __name__ == "__main__":
    main()
